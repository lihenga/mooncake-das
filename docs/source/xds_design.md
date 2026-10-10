# 基于 xDS 的 DFS KV Cache 直接预取设计

## 0. 文档状态与阅读指南

- 状态：实现规格（Implementation-ready Design）
- 目标组件：SGLang Mooncake Direct Linker、Mooncake Store、DistributedStorageBackend
- 首个可合并版本：从 BUCKET allocator 的 DFS 文件读取对齐的 KV Cache range，直接写入海光 DCU 显存
- 业务写路径：首版不实现 xDS 写，继续使用现有 Mooncake DFS 写路径
- xDS 运行时：海光 DTK/xDS、`libhyfile.so`
- DFS 首期范围：经过真实 round-trip preflight 的 Parastor/HYFS mount
- 对齐要求：首版固定 4096 字节，不提供生产覆盖开关
- 当前代码校验基线：
  - `mooncake-das`：`feat/optimize_prefetch`，commit `3292232f2a296409fe3bf88096b31e7fa037df05`
  - `sglang-das`：`feature/optimize_the_direct_linker`，commit `c4257ce7b190710b953504645aea1cdd7a53035e`

本文是代码实现依据。实现若需要改变本文标为“首版契约”的行为，必须先更新设计和对应测试，不得通过隐式 fallback 或未记录的兼容逻辑改变语义。

### 阅读指南

| 阅读目的                               | 重点章节                           |
| ---------------------------------- | ------------------------------ |
| 快速理解做什么、为什么做                       | 1.1～1.4、2.1、10.4               |
| 实现 Phase 0～2 的核心代码                 | 2～7、9.1～9.3                    |
| 实现 `RealClient` direct/fallback 路由 | 3.3～6.1                        |
| 实现 `libhyfile.so` shim/backend     | 2.2、3.2、5.1、6、7.1              |
| 实现 BUCKET-first MVP                | 1.3.5、4.2.3～4.2.4、5、6、9.1      |
| 编写测试和验收                            | 8、10.1、10.4                    |
| 只关注后续优化                            | 5.3.2、7.2.2、9.1 中的 Phase 2.5～4 |

本次 Phase 0～2 直接相关的实现核心是第 3～7 章；第 1～2 章定义背景、范围和不可违反的契约，第 8～10 章定义观测、测试、实施和验收。标明“后续”的 GPU bounce、业务 xDS write、异步/batch 以及可选 SHARD compatibility 不阻塞 BUCKET MVP。

## 1. 背景、现状与目标

### 1.1 摘要

SGLang 使用 Mooncake 作为 L3 KV Cache。KV 对象位于 DFS 时，当前 range-get 路径即使只需要对象的一部分，也会把完整对象读入 Mooncake pinned host arena，再异步 H2D scatter 到 SGLang 的 DCU KV pool。

本文引入一条 xDS/HyFile 直接读取路径：Mooncake 把 DFS 文件的绝对 offset、已注册 DCU KV pool 的原始基址以及目标 pool-relative offset 传给 `hipFileRead`，使满足条件的 KV range 从 DFS 直接进入最终 DCU 地址。

首版保持以下原则：

1. 不改变 SGLang 公开 KV connector API。
2. 不改变 Master replica 元数据和分配协议。
3. 每个 key 在一次调用中选择全 direct 或全 fallback。
4. xDS direct range 不分配 Mooncake host staging，也不执行 H2D。
5. xDS 不可用或请求不适用时，安全回退现有路径。
6. xDS 已接触目标显存后，只有确认 DMA 已停止，才允许整 key fallback【*某个range读失败，则整体回退】。
7. 当前 Mooncake 代码没有独立的 DFS/xDS worker 进程。RealClient、Client、DistributedStorageBackend 以及 BUCKET BatchRead 使用的线程池都位于同一个承载进程内；停止某个 backend 线程、线程池任务或 RPC handler 不能终止已经提交的设备 DMA。若 xDS read 已提交但无法证明 DMA 已停止，首版必须禁止 host fallback，并禁止释放、覆盖或复用对应的目标 KV block；同时将 xDS backend 和当前承载进程标记为 fatal/unhealthy，由外部 supervisor 终止并重启整个承载进程（没自动拉起，就人为操作重启）。SGLang 内嵌模式下，该进程是承载 RealClient 的 SGLang worker 进程；独立 RPC 模式下，该进程是 mooncake_client。新进程启动后必须重新初始化 xDS driver、file handle、buffer registration、连接和 session。只有未来将 xDS 放入独立 helper process 后，该 helper process 才能成为单独的故障隔离和重启边界。
8. 生产数据面和 preflight 都只使用 xDS read；preflight 通过 POSIX 写入已知数据，再用 `hipFileRead` 读入 DCU 并校验。

本文所称“零拷贝”仅表示 Mooncake 不分配 host staging buffer，也不显式执行 H2D payload copy。hipFileRead 返回成功并不能证明 libhyfile.so、DFS client 或内核驱动内部未使用临时 host/device bounce buffer。除非厂商提供可区分 direct-DMA bytes 与 bounce bytes 的可信计数器、trace 或正式语义保证，本文只将该路径称为“Mooncake 零 host staging”，不宣称端到端完全零拷贝。。

### 1.2 术语

| 术语                 | 含义                                           |
| ------------------ | -------------------------------------------- |
| xDS                | 海光 DCU 软件栈提供的 GDS 类直接存储能力                    |
| HyFile             | 本设计使用的 xDS 用户态接口，符号以 `hipFile*` 命名           |
| direct range       | 通过 `hipFileRead` 直接写入最终 DCU KV 地址的 range     |
| GPU bounce         | 先读入已注册的对齐 DCU bounce buffer，再 D2D 到最终地址      |
| host fallback      | 当前 DFS → pinned host arena → H2D 路径          |
| registration owner | 负责最终 deregister 的唯一、move-only 资源所有者          |
| registration pin   | 一次 I/O 持有的共享引用，阻止 owner 在 I/O 完成前 deregister |
| session generation | 区分同一 key 的新旧 get session 的单调递增编号             |
| publish            | 将一次 range-get 对外报告为成功，使 SGLang 可以消费目标 KV     |
| target state       | xDS 失败后，目标显存是否未触碰、可安全覆盖或状态未知                 |

### 1.3 当前实现基线

#### 1.3.1 SGLang 调用方式

SGLang 初始化 Mooncake Direct Linker 时枚举 KV pool 的底层 tensor storage，并调用：

```python
store.register_buffer(storage.data_ptr(), storage.nbytes())
```

加载 KV 时，SGLang 按 layer 或整 page 构造：

```text
key
目标 DCU 地址 dst
range 长度 length
对象内源偏移 object_offset
```

然后调用：

```python
batch_get_into_multi_buffer_ranges(keys, ptrs, sizes, offsets)
```

公开返回语义为逐 key 返回实际读取字节数；负数表示错误。调用是同步的：返回成功前，目标 range 必须已经可安全消费。

#### 1.3.2 当前 DFS 读取路径

当前 DFS replica 的 range-get 路径如下：

```text
SGLang DCU KV ranges
        ^
        | DfsAsyncScatterContext: async H2D + stream synchronize
        |
Mooncake pinned host arena（cache miss 时每个 key 覆盖完整对象）
        ^
        | BatchGet / DistributedStorageBackend::BatchRead
        |
POSIX buffered 或 O_DIRECT read
        ^
        |
Distributed filesystem
```

具体行为：

1. `batch_get_session_start` 查询并缓存一个完整 replica 和 lease。
2. `batch_get_into_multi_buffer_ranges` 校验参数、session、lease 和对象边界。
3. memory replica 走现有 Transfer Engine range read。
4. DFS replica 先查询 `get_session_object_cache_`。
5. session cache miss 时，为完整对象分配 aligned pinned host arena view。
6. `Client::BatchGet` 构造现有 `DfsReadRequest`，由 backend 读取完整对象。
7. `DfsAsyncScatterContext` 合并相邻 H2D copy，并同步使用过的 stream。
8. 完成后填写逐 key 字节数并返回。

当前校验基线中不存在 Mooncake `DfsPrefetcher` 或 `TryConsume`。本文不把不存在的 prefetcher 纳入首版路由。SGLang 业务层所称“prefetch”指 L3 KV 恢复行为，不等同于 Mooncake 内部 host-object prefetcher。

#### 1.3.3 当前 POSIX direct-read 的准确语义

`MOONCAKE_DFS_DIRECT_READ_ENABLED` 当前只控制 BUCKET 读取是否优先使用 direct handle；当前配置默认值为 `true`。SHARD 的 `BatchReadShard` 始终通过普通 `ReadAt` 读取 `object_size`，不会因该开关切换到 `DirectReadAt`。

BUCKET 开启该开关时：

1. `OpenFileDirect` 优先以 `O_RDONLY | O_DIRECT | O_CLOEXEC` 打开独立 fd；仅在文件系统返回 `EINVAL`、`EOPNOTSUPP` 或 `ENOTSUP` 时透明退回 `O_RDONLY | O_CLOEXEC` 的 buffered fd。因此代码中的“direct handle”不保证底层 fd 一定实际带有 `O_DIRECT`。
2. backend 仍读取完整 `descriptor.aligned_size`，包括 value 后的 padding，而不是只读取 SGLang 请求的 ranges。
3. `DirectReadAt` 只有在文件 offset、每个非空 iovec 的基址和长度均满足 4096 字节对齐时，才直接 `preadv` 到调用方提供的 host slices。
4. 任一对齐条件不满足时，adapter 计算覆盖请求的对齐窗口，使用池化或临时 aligned host staging 执行 `pread`，再 `memcpy` 到调用方 slices。
5. 即使 adapter 走无 bounce 的对齐分支，range cache miss 的上层路径仍然分配完整对象 host arena，并在 DFS 读取后执行 H2D scatter。

因此 xDS 的主要收益是消除完整对象 host arena、读取放大和后续 H2D payload copy，不能笼统描述为“消除每次 POSIX O_DIRECT 的 adapter bounce”。

#### 1.3.4 当前 DFS 写路径

当前 device source 的 DFS 写路径是：

```text
SGLang DCU KV
    |
    | D2H
    v
Mooncake owned pinned host staging
    |
    | DistributedStorageBackend::BatchWrite
    v
DFS
```

同步 SHARD 写和异步 BUCKET 写都会先把 device slice 复制到 Mooncake 所有的 host staging，以保证调用方返回后源 DCU buffer 被复用时，后台 DFS 写仍然安全。

首版 xDS 设计不改变该业务写路径。

#### 1.3.5 当前 SHARD 与 BUCKET 布局

SHARD descriptor 的 `offset` 指向 allocator 分配的 value 起点。

当前 BUCKET entry 的数据文件布局是：

```text
offset       = AlignUp(previous_entry_end, alignment)
[value bytes][trailing padding]
object_size  = value.size()
aligned_size = AlignUp(object_size, alignment)
```

因此当前 BUCKET 的 `descriptor.offset` 已经是对齐后的 value 起点。BUCKET 是否可以 direct，取决于：

```text
descriptor.offset + object_offset
dst
length
```

三者是否满足 xDS 对齐约束，而不是取决于 header/key 前缀。首版不修改 BUCKET 磁盘格式，也不引入 layout v2。

### 1.4 问题与目标

#### 1.4.1 当前路径的主要成本

1. SGLang 只请求部分 range 时，Mooncake 仍读取完整对象。
2. cache miss 时为完整对象申请 pinned host arena。
3. DFS → host 和 host → DCU 形成两段数据搬运。
4. 当前 PosixFsAdapter::DirectReadAt 只有在文件 offset、每个 iovec 的地址与长度均满足 direct-I/O 对齐要求时，才直接调用 preadv。如果任一条件不满足，adapter 会把覆盖请求范围的对齐窗口读取到临时 host staging buffer，再复制到调用方提供的 host iovec；这属于 POSIX fallback 路径的 host bounce。
5. H2D scatter 即使合并并异步提交，仍占用 host bandwidth 和 DCU copy engine。
6. 大对象、小 range 场景存在显著 read amplification。

#### 1.4.2 目标

1. 对满足条件的 DFS range 使用 xDS 直接读入最终 DCU 地址。
2. direct path 只读取请求 range，不强制读取完整对象。
3. direct path 不申请 Mooncake host staging，不提交 H2D。
4. 复用 SGLang KV pool 的长生命周期注册，不逐请求注册显存。
5. 保持现有 Python API、session lease 和逐 key 返回码兼容。
6. xDS 缺失、不可用或请求不适用时，回退现有 POSIX 路径。
7. 通过明确的 target state 保证部分写入后的 fallback 安全。
8. 支持 `posix`、`xds-auto` 和 `xds-required` 三种运行模式。
9. 支持 metrics、trace、故障注入、灰度和一键关闭。
10. 在真实 workload 上证明端到端收益，而不仅是 `hipFileRead` 带宽。

#### 1.4.3 非目标

- 首版不实现业务 DCU → DFS xDS 写。
- 不改变 Master replica 元数据协议和 allocator 分配协议。
- 不改变 SGLang 公开 KV connector API。
- 不实现 HyFile batch/event/poll/async API。
- 不在进程内强制取消已进入 `hipFileRead` 的调用。
- 不允许为了启用 xDS 绕过 descriptor、路径、lease 或生命周期校验。
- 不把 `libhyfile.so` 打包进 Mooncake wheel。
- 不承诺其他 DFS、FUSE mount 或未知 DTK 版本可用。
- 不承诺厂商驱动内部绝无 bounce。

## 2. 首版设计原则与依赖

### 2.1 首版关键决策

#### 2.1.1 新增 range-aware DFS API，不改变旧 API

现有 `DfsReadRequest` 保持“从对象 value 起点开始顺序填满 host slices”的语义，继续服务整对象读取和 fallback。

新增独立请求：

```cpp
struct DfsReadRange {
    void* dst = nullptr;
    uint64_t length = 0;
    uint64_t object_offset = 0;
    uint64_t registration_id = 0;
    int32_t device_id = -1;
};

struct DfsRangeReadRequest {
    std::string key;
    DistributedFSDescriptor descriptor;
    std::vector<DfsReadRange> ranges;
};
```

新增内部方法：

```cpp
std::vector<tl::expected<void, XdsError>>
Client::BatchReadDfsRanges(
    const std::vector<DfsRangeReadRequest>& requests);

std::vector<tl::expected<void, XdsError>>
DistributedStorageBackend::BatchReadRanges(
    const std::vector<DfsRangeReadRequest>& requests);
```

这两个方法：

- 不重新选择 replica；
- 不接受缺少 key/descriptor 的请求；
- 不经过 `TransferSubmitter`；
- backend 对每个 key 聚合其全部 direct op，只有全部 exact-length 且完成同步才返回成功；
- 返回结果必须与各自输入等长、同序；内部排序和 coalescing 不得改变外部结果顺序；
- 只返回逐 key 成功或带 `target_state` 的失败，不计算公开 API 的业务成功字节数；
- 记录 xDS/fallback I/O metrics。

`AcceleratorFileIo::Read` 仍返回底层实际完成字节数，用于强制校验 `actual == requested`；这个字节数是单个物理 direct op 的完成信息，不直接作为 Python API 返回值。`RealClient` 是唯一计算公开逐 key 成功字节数的层：它保留现有 `SessionRangeReadRequest::original_idx`，为 direct/fallback 子计划维护局部 `request_index -> original_idx` 映射，并且仅在该 key 的全部 ranges、设备同步和发布检查均成功后返回 `sum(all_sizes[original_idx])`。`DfsRangeReadRequest` 不携带公开请求下标，backend 不依赖调用方的结果数组布局。

子层返回数量与输入数量不一致时，对应子批次全部返回 `INTERNAL_ERROR`，不得按 key 猜测映射。由于公开返回类型当前为 `int`，planner 还必须在任何 I/O 前验证每 key 请求字节总和不超过 `INT_MAX`。

#### 2.1.2 每 key 单一路径

首版一个 key 中只要有任一 range 不满足 direct 条件，整个 key 走 host fallback。

禁止在一次调用中对同一个 key 同时执行 xDS 和 H2D。xDS 失败后允许 fallback 时，也必须覆盖该 key 的全部 ranges。

#### 2.1.3 `AcceleratorFileIo::Read` 是原子安全边界

一次 `Read` 内部负责：

```text
PID/generation/边界复核
→ DeviceGuard / hipSetDevice
→ 获取进程级 HyFile mutex
→ hipFileRead
→ exact-length 校验
→ hipDeviceSynchronize
→ 生成稳定错误分类和 target state
→ 释放 HyFile mutex
```

上层不得直接调用 vendor synchronize，也不得直接解释 vendor 返回码。

#### 2.1.4 首版同步和串行

首版：

- 所有 HyFile API 调用由一个进程级 mutex 串行保护；
- `hipFileRead` 和其后的 `hipDeviceSynchronize` 在同一次加锁范围内完成；
- 不提供关闭串行保护的配置；
- 不假设 SGLang compute stream 与 HyFile DMA 自动有序；
- 不使用未确认的 batch/event/poll 结构体或函数。

#### 2.1.5 fatal 状态 fail-stop

如果 `hipDeviceSynchronize` 失败，无法证明驱动不再访问目标显存。首版行为固定为：

```text
标记 backend fatal
→ 拒绝新 xDS 请求
→ 当前请求不 fallback、不发布成功
→ 当前 worker fail-stop
→ 外部 supervisor 重启 worker
```

首版不实现 SGLang KV allocator 的 poisoned-block quarantine 协议。

### 2.2 依赖和兼容性

#### 2.2.1 运行时依赖

- 海光 DCU；
- 海光 DTK/ROCm 用户态运行时；
- 匹配的内核驱动和固件；
- `libhyfile.so`；
- Parastor/HYFS client 和受支持的 mount；
- 支持 `O_DIRECT` 的路径；
- HIP device/context 和 synchronize API；
- 能重启失效 worker 的外部 supervisor/watchdog。

默认动态库路径：

```text
/opt/hyhal/lib/libhyfile.so
```

Mooncake 使用显式路径 `dlopen`，不做 soname 自动协商，不把动态库复制进 wheel。

#### 2.2.2 必需和可选符号

首版 required symbols：

```text
hipFileDriverOpen
hipFileDriverClose_v2
hipFileHandleRegister
hipFileHandleDeregister
hipFileBufRegister
hipFileBufDeregister
hipFileRead
```

preflight 使用 POSIX 写入已知测试数据，再通过 `hipFileRead` 读入 DCU
buffer 并回读校验；首版不要求也不加载 `hipFileWrite`。

可选加载但首版不调用：

```text
hipFileDriverSetMaxCacheSize
hipFileDriverSetMaxPinnedMemSize
```

任一 required symbol 缺失时：

- `xds-auto`：禁用 xDS，继续 POSIX；
- `xds-required`：backend 初始化失败。

#### 2.2.3 ABI 规则

首版只维护一套最小手工 ABI 声明，不依赖或探测 `hipfile.h`。ABI 声明集中在
`hyfile_abi.h`，生产实现只能通过该声明和 `dlopen`/`dlsym` 调用
`libhyfile.so`，其他编译单元不得重复声明 vendor 类型或函数。

规则如下：

1. 只声明首版 read-only 路径实际使用的类型、常量和 7 个 required symbols；
2. 对错误结构体和文件描述符执行编译期 size、alignment、offset 校验；
3. required symbol 缺失时 fail closed，但不做 build-id/hash 强制拒绝；
4. 部署清单仍必须记录 DTK、driver 和 `libhyfile.so` 版本/build-id，便于定位兼容性问题；
5. 启用业务流量前必须执行真实 mount、逐 device 的已知数据读回 preflight；
6. 错误码映射以已确认的当前 ABI 为准，未知错误按保守策略处理。

在厂商确认前，以下问题是生产 backend 的阻塞项：

- `hipFileRead` 的精确 C 原型；
- 错误是 `-errno`、`-1 + errno` 还是独立 HyFile code；
- 返回后 DMA 是否已经停止；
- `hipDeviceSynchronize` 是否为必要且充分的安全同步点；
- fork、context 和多 device 行为。

#### 2.2.4 支持矩阵

部署必须记录：

```text
DCU 型号
DTK 版本
libhyfile.so 版本/build-id
ROCm/libgalaxyhip 用户态版本
内核驱动版本
固件版本
Linux kernel
Parastor/HYFS client 版本
mount 类型和参数
```

## 3. 架构、核心类型与生命周期

### 3.1 总体架构

```text
SGLang Mooncake Direct Linker
  - KV pool lifetime
  - register/unregister buffers
  - batch_get_into_multi_buffer_ranges
                  |
                  v
RealClient
  - session generation/pin
  - session cache routing
  - batch target-overlap validation
  - registration interval pinning
  - per-key direct/fallback classification
                  |
                  v
Client::BatchReadDfsRanges
                  |
                  v
DistributedStorageBackend
  - descriptor validation
  - SHARD/BUCKET target resolution
  - value/file offset calculation
  - range coalescing
  - xDS file-cache pinning
          |                         |
          v                         v
AcceleratorFileIo              FileSystemAdapter
  HyFileBackend                 existing fallback
  libhyfile shim                full object → host
          |                         |
          v                         v
    xDS DMA → DCU              async H2D scatter
```

`FileSystemAdapter` 继续负责普通文件系统语义。`AcceleratorFileIo` 只负责 vendor direct-device I/O，不负责 replica 选择、session 或 allocator layout。

### 3.2 核心类型和所有权

#### 3.2.1 能力快照

```cpp
struct DirectIoCapabilities {
    bool available = false;
    bool supports_async = false;            // 首版 false
    bool supports_scatter_gather = false;   // 首版 false
    bool supports_distributed_fs = false;
    uint64_t file_offset_alignment = 4096;
    uint64_t device_address_alignment = 4096;
    uint64_t length_alignment = 4096;
};

enum class DeviceXdsState {
    kUntested,
    kAvailable,
    kUnavailable,
    kFatal,
};
```

动态库、driver 和 DFS mount 兼容性属于 backend 级状态；preflight 结果、可用性和失败原因属于 per-device 状态。backend 维护 `device_id -> DeviceXdsState`，不能用一个全局 `capabilities.available` 代表所有 device。能力在对应 device 完成初始化和 preflight 后冻结，I/O 热路径不重新 probe。首版 alignment 来自经过验证的部署 ABI，固定为 4096，不允许普通用户覆盖。

#### 3.2.2 错误和目标状态

```cpp
enum class XdsErrorClass {
    kNotEligible,        // 驱动调用前发现不适用，可走 fallback
    kContractViolation, // 调用方或内部状态违反契约，不盲目 fallback
    kTransientIo,       // 可能恢复的 I/O 失败，结合 target_state 处理
    kBackendFatal,      // driver/context/sync 不可信
};

enum class XdsTargetState {
    kUntouched,
    kStoppedSafeToOverwrite,
    kUnknownMayStillBeWritten,
};

struct XdsError {
    XdsErrorClass error_class;
    XdsTargetState target_state;
    int64_t raw_code;
    std::string operation;
};

struct DirectReadCompletion {
    uint64_t bytes;
};
```

#### 3.2.3 owner 与 pin 分离

资源类型分为：

```text
RegisteredFileOwner     唯一、move-only，负责最终 handle deregister + fd close
RegisteredFilePin       可复制的在途 keepalive，不负责清理

RegisteredBufferOwner   唯一、move-only，负责最终 buffer deregister
RegisteredBufferPin     可复制的在途 keepalive，携带 generation/bounds/device
```

`DirectReadOp` 只持有 pin，不持有 owner：

```cpp
struct DirectReadOp {
    RegisteredFilePin file;
    RegisteredBufferPin buffer;

    uint64_t file_offset = 0;
    void* registered_base = nullptr;
    uint64_t device_offset = 0;
    uint64_t length = 0;
    int32_t device_id = -1;

    size_t request_index = 0;
    size_t first_range_index = 0;
    size_t covered_range_count = 1;
};
```

必须满足：

```text
registered_base == hipFileBufRegister 使用的原始 base
registered_base + device_offset == 最终 dst
dst + length 完全位于 registration 内
pin generation 与 registry 当前 generation 相同
```

#### 3.2.4 `AcceleratorFileIo` 接口

```cpp
class AcceleratorFileIo {
  public:
    virtual ~AcceleratorFileIo() = default;

    virtual DirectIoCapabilities Capabilities() const = 0;
    virtual DeviceXdsState DeviceState(int32_t device_id) const = 0;

    virtual tl::expected<RegisteredFileOwner, XdsError> RegisterFile(
        UniqueFd direct_fd, const FileIdentity& identity) = 0;

    virtual tl::expected<RegisteredBufferOwner, XdsError> RegisterBuffer(
        void* base, size_t length, int32_t device_id) = 0;

    virtual tl::expected<DirectReadCompletion, XdsError> Read(
        const DirectReadOp& op) = 0;

    virtual tl::expected<void, XdsError> DeregisterFile(
        RegisteredFileOwner&& owner) = 0;

    virtual tl::expected<void, XdsError> DeregisterBuffer(
        RegisteredBufferOwner&& owner) = 0;
};
```

`UniqueFd` 的所有权在 `RegisterFile` 成功时转移给 `RegisteredFileOwner`；失败时由调用方或返回对象保证关闭，不允许泄漏或双重关闭。

### 3.3 Buffer registry

#### 3.3.1 entry 数据

```cpp
enum class RegistrationState {
    kRegistering,
    kActive,
    kDraining,
    kQuarantined,
};

enum class XdsIneligibleReason {
    kNone,
    kModeDisabled,
    kUnsupportedMemoryKind,
    kDeviceUnavailable,
    kRegistrationFailedSafe,
};

struct RegisteredDeviceRegion {
    uintptr_t base;
    size_t length;
    MemoryKind memory_kind;
    int32_t device_id;
    pid_t pid;
    uint64_t generation;
    RegistrationState state;
    uint64_t logical_refcount;
    uint64_t inflight_refcount;
    TransferEngineRegistration transfer_registration;
    std::optional<RegisteredBufferOwner> xds_owner;
    bool xds_eligible;
    XdsIneligibleReason xds_ineligible_reason;
};
```

使用有序 interval registry，而不是热路径线性扫描所有 base。

#### 3.3.2 注册规则

公开接口保持 `register_buffer(base, size)`。Mooncake 在注册阶段允许查询一次 pointer/device 属性，得到实际 memory kind 和 device id；热路径不查询。公开注册成功表示该区域至少可继续服务现有 Transfer Engine/host fallback 路径，不表示它一定具备 xDS 能力。

规则：

- `base == nullptr` 或 `size == 0`：拒绝；
- `base + size` 溢出：拒绝；
- 部分重叠已有区域：拒绝；
- 相同 base、size、device、PID 的重复注册：只增加 logical refcount；
- 相同 base 但 size/device 不同：contract violation；
- fork 后 PID 改变：继承 entry 一律不可用；
- xDS 注册成功前不得发布为 direct eligible。

通用事务顺序：

```text
validate interval/memory kind/device/PID
→ state = registering
→ Transfer Engine register（需要时）
→ 查询对应 device 的 xDS 状态
→ 可用时 hipSetDevice + hipFileBufRegister
→ 按运行模式发布 composite 或 TE-only entry
```

各模式行为固定为：

| 模式 | 单个 buffer 的 xDS 注册失败行为 |
|---|---|
| `posix` | 不调用 HyFile，发布 TE-only active entry。 |
| `xds-auto` | 若 xDS 未创建残留资源，或失败后的 deregister/回滚已确定成功，则发布 TE-only active entry，`xds_owner=nullopt`、`xds_eligible=false`，并记录结构化原因；`register_buffer` 对调用方仍返回成功。 |
| `xds-required` | 逆序回滚 Transfer Engine registration，`register_buffer` 返回失败，不发布 entry。 |

不支持的 memory kind、对应 device 未通过 preflight，以及确定失败且已安全清理的 `hipFileBufRegister`，在 `xds-auto` 下均属于 TE-only 降级。不得在每次 range read 时重新尝试临时注册；恢复 xDS eligibility 必须经过显式重新注册或 backend/device 重建。

如果 `hipFileBufRegister` 可能部分成功，而后续 deregister/回滚失败或无法证明 driver 不再引用该显存，则任何模式都不得发布 TE-only entry。entry 进入 `kQuarantined`，`register_buffer` 返回失败，调用方必须保持 storage 存活，首版使 worker fail-stop 并由外部 supervisor 重建。

#### 3.3.3 pin

planner 查找 `[dst, dst + length)` 时：

1. entry 必须为 `kActive`；
2. range 必须完全位于一个 entry；
3. direct plan 还要求 `xds_eligible=true` 且 `xds_owner` 存在；TE-only entry 只能进入现有 fallback 路径；
4. PID 和 generation 必须匹配；
5. 增加 inflight refcount；
6. 返回 `RegisteredBufferPin`。

pin 生命周期覆盖：

- xDS read；
- read 后 device synchronize；
- 必要的 GPU bounce D2D；
- 错误分类；
- publish 或 fallback 决策。

#### 3.3.4 解除注册

```text
logical_refcount--
→ 非零：返回成功
→ 归零：state = draining
→ 从 active interval lookup 中移除
→ 等待 inflight_refcount == 0
→ xds_owner 存在时 hipSetDevice + hipFileBufDeregister(base)
→ Transfer Engine deregister
→ 删除 entry
```

TE-only entry 跳过 HyFile deregister。`xds_owner` 存在且 `hipFileBufDeregister` 失败时，entry 进入 `kQuarantined`。运行期不得把同一地址重新注册为正常可用。首版生产策略为使 worker 退出，避免 Python/PyTorch 释放仍可能被驱动引用的显存。

#### 3.3.5 SGLang 生命周期

SGLang Direct Linker 保存成功注册的 `(base, size)` 列表。关闭顺序：

```text
停止接收新 load/offload
→ join load/offload/read-plan worker
→ 结束所有 session
→ 对成功注册的 buffer 调用 unregister_buffer
→ storage.close()
→ Direct Linker/pool 对象才可析构
```

Mooncake `close()` 同时执行 defensive drain，清理调用方遗漏的 registration，但这不替代 SGLang 的显式顺序。

### 3.4 Session 和可见性

#### 3.4.1 session state

同一 key 的每次 `batch_get_session_start` 创建新的内部 generation：

```cpp
struct GetSessionState {
    std::string key;
    uint64_t generation;
    QueryResult query_result;
    time_point lease_deadline;
    bool closing;
    uint64_t inflight_reads;
};

struct SessionReadPin {
    std::shared_ptr<GetSessionState> state;
    uint64_t generation;
};
```

#### 3.4.2 提交前检查

创建 `SessionReadPin` 时检查：

- session 存在；
- replica 完整且类型受支持；
- lease 未过期；
- `closing == false`；
- range 不越过对象边界。

#### 3.4.3 完成前发布检查

direct、session-cache H2D 和 host fallback 完成后，在填写成功返回码前再次检查：

```text
当前 key 仍指向同一个 shared session state
generation 未改变
closing == false
lease 未过期
所有设备操作已完成
```

失败时不发布成功。只要 device synchronize/H2D synchronize 已成功，目标内存可由上层按失败路径安全回收或覆盖。

#### 3.4.4 session end

`batch_get_session_end`：

1. 在 `session_mutex_` 下标记 `closing=true`；
2. 从当前 session map 中移除；
3. 删除 session object cache map entry；
4. 不持锁等待可能阻塞的 I/O；
5. in-flight `SessionReadPin` 保持 state 和 cache source owner 存活；
6. 旧 generation 的完成结果不得发布给之后的新 session。

Session 生命周期与 KV pool xDS registration 生命周期独立。session end 不等于可以 deregister KV pool。

## 4. 请求规划与文件定位

### 4.1 请求验证和 planner

#### 4.1.1 batch 级验证

提交任何 I/O 前完成：

- `keys`、`all_buffers`、`all_sizes`、`all_src_offsets` 数量一致；
- 每个 key 至少有一个 range；
- 零长度 range 非法；
- 非零 range 的 `dst` 不为空；
- `object_offset + length` 无溢出且不越过 object size；
- batch 中 key 必须唯一；首版不隐式合并重复 key；
- 所有目标 `[dst, dst + length)` 不溢出；
- 整个 batch 的目标地址区间不得有任何重叠，包括同 key 完全重复 range；
- request index 与结果 index 一一对应。

目标区间冲突在调用任何 xDS、POSIX 或 H2D 前返回 `INVALID_PARAMS`，避免不同 producer 竞争。

#### 4.1.2 路由顺序

```text
创建 SessionReadPin 并校验 ranges
  → session object cache 命中？
      是：加入 host-source H2D plan
      否：
  → 整 key direct eligibility
      全部满足：加入 direct plan
      任一不满足：加入 initial fallback plan
```

首版没有 `DfsPrefetcher` 分支。

#### 4.1.3 direct eligibility

同一 key 的全部 range 必须同时满足：

```text
xDS backend 基础能力可用
request device 的 DeviceXdsState == kAvailable
capabilities.supports_distributed_fs
range 完全位于 active registration
registration.xds_eligible 且 xds_owner 存在
registration PID/generation 有效
registration device 与 request device 一致
同一 key 的 ranges 属于同一 device（首版限制）
file_offset = value_offset + object_offset 无溢出
file_offset % 4096 == 0
uintptr(dst) % 4096 == 0
length % 4096 == 0
0 < length <= SSIZE_MAX
coalescing 后每个 DirectReadOp.length >= MIN_READ_SIZE
circuit breaker 允许提交
```

普通 eligibility 不满足属于 `kNotEligible`，直接加入 host fallback，不调用 HyFile，不计 backend 故障。

#### 4.1.4 range coalescing

首版只合并 request 中相邻、同时满足以下条件的 range：

```text
相同 file pin
相同 buffer registration pin
相同 device
前一 file range 末尾 == 后一 file range 起点
前一 device range 末尾 == 后一 device range 起点
合并后 length <= SSIZE_MAX
合并后仍位于同一对象和 registration 内
```

不为了合并而跨对象读取，不读取相邻 key 数据，不重排可能改变覆盖关系的 range。

#### 4.1.5 planner-only telemetry

在真实 backend 上线前提供 dry-run 模式：执行完整 planner，但所有读取仍走现有 fallback。至少记录：

```text
eligible keys/ranges/bytes
coalesced DirectReadOp 数量
file-offset unaligned
device-address unaligned
length unaligned
unregistered range
mixed-device key
below-min-read
目标区间冲突
full-object read amplification
预计 device synchronize 次数
```

该数据决定是否进入真实 HyFile 接入阶段。

### 4.2 文件和 allocator 地址解析

#### 4.2.1 通用规则

上层不得自行拼接文件路径、bucket 文件名或 shard offset。所有请求先通过 `DistributedStorageBackend::ResolveTarget` 的等价验证。

range 文件偏移统一为：

```text
file_offset = resolved_value_offset + object_offset
```

#### 4.2.2 SHARD

- `descriptor.shard_idx` 必须落在初始化 shard 表内；
- `descriptor.file_path` 必须与固定 shard path 一致；
- `descriptor.offset/object_size/aligned_size` 必须落在 shard capacity 内；
- 首版 xDS direct fd 在 backend 初始化阶段以 `O_RDONLY | O_DIRECT | O_CLOEXEC` 打开并注册；
- `resolved_value_offset = descriptor.offset`。

#### 4.2.3 BUCKET

- `descriptor.shard_idx` 被解释为 bucket id，必须非负并可安全格式化；
- backend 在 `canonical_root_dir` 下根据 bucket id 重建预期 data path；
- 不直接信任 descriptor 中的任意路径打开文件；
- 使用当前 `BucketEntryLayout` 验证 offset、object size、aligned size 和 bucket capacity；
- `resolved_value_offset = descriptor.offset`；
- BUCKET 文件按需注册并放入有界 LRU xDS file cache。

#### 4.2.4 File identity 和 cache

`FileIdentity` 至少包含：

```text
allocator type
shard index 或 bucket id
canonical reconstructed path
st_dev
st_ino
可用时的 generation/version
```

禁止仅以 POSIX fd 数字作为 identity，因为 fd 会被复用。

xDS direct fd 与 fallback fd 分离：

```text
xDS:      O_RDONLY | O_DIRECT | O_CLOEXEC
fallback: 现有 buffered/direct adapter 自己管理的 fd
```

关闭顺序：

```text
从 cache 摘除
→ 禁止新 pin
→ 等待 file inflight pin 归零
→ hipFileHandleDeregister
→ close(direct_fd)
```

SHARD owner 在 backend shutdown 时清理。BUCKET LRU eviction 使用同样顺序。

## 5. xDS 调用与完整读取流程

### 5.1 HyFile 调用契约

#### 5.1.1 文件注册

```text
open(path, O_RDONLY | O_DIRECT | O_CLOEXEC)
→ fstat 并构造 FileIdentity
→ hipFileHandleRegister(
       out_handle,
       type = HIP_FILE_HANDLE_TYPE_OPAQUE_FD,
       fd = direct_fd,
       fs_ops = nullptr)
```

成功后 vendor handle 与 fd 由 `RegisteredFileOwner` 共同持有。

#### 5.1.2 buffer 注册

```text
hipSetDevice(device_id)
→ hipFileBufRegister(
       base,
       length,
       HIP_FILE_RDMA_REGISTER)
```

首版固定 `HIP_FILE_RDMA_REGISTER = 1`。不使用 relaxed ordering flag。

#### 5.1.3 direct read 参数映射

对于 `(dst, length, object_offset)`：

```text
file_offset   = resolved_value_offset + object_offset
base          = registration.base
device_offset = uintptr(dst) - uintptr(base)
bytes         = length
```

语义调用：

```cpp
DeviceGuard guard(op.device_id);
std::lock_guard<std::mutex> lock(hyfile_api_mutex);

ssize_t ret = api.hipFileRead(
    op.file.vendor_handle(),
    op.registered_base,
    op.length,
    op.file_offset,
    op.device_offset);
```

精确原型固定在 `hyfile_abi.h` 的手工 ABI 声明中；以上代码表示参数语义。

#### 5.1.4 成功和同步

只有 `ret == requested_length` 才进入成功候选。随后必须在同一 device context、同一 HyFile mutex 范围内执行 `hipDeviceSynchronize()`。

```text
exact-length + synchronize success
    → DirectReadCompletion{length}

exact-length + synchronize failure
    → kBackendFatal + kUnknownMayStillBeWritten
```

#### 5.1.5 非成功 read

任何短读或 vendor error 都视为失败。保持 file pin、buffer pin 和 session pin，先执行 `hipDeviceSynchronize()`：

```text
synchronize success
    → 根据已确认的手工 ABI 映射错误类别
    → target_state = kStoppedSafeToOverwrite

synchronize failure
    → kBackendFatal
    → target_state = kUnknownMayStillBeWritten
    → worker fail-stop
```

首版不在 HyFile adapter 层自动重试 `hipFileRead`，避免目标状态不清时重复写入。

### 5.2 完整读取执行流程

#### 5.2.1 初始化

```text
解析配置
→ dlopen/dlsym libhyfile.so
→ 记录部署版本并验证 required symbols
→ hipFileDriverOpen
→ 对每个目标 device 执行 round-trip preflight
→ 初始化 buffer registry 和 file cache
→ 发布 backend 基础能力和逐 device DeviceXdsState
```

driver open、required symbols 或目标 mount 的公共检查失败属于 backend 级失败。单个 device 的 probe 失败只更新该 device 的状态；是否继续启动由 `xds-auto`/`xds-required` 模式决定，详见 7.1.3。

#### 5.2.2 一次 batch

首版为便于证明安全，执行顺序固定为：

1. 校验所有参数、session、对象边界和 batch 目标地址无重叠。
2. 查询 session object cache，为命中项构建 host-source H2D plan，但暂不提交。
3. 对 cache miss 做 direct eligibility 和 coalescing。
4. pin 所有 direct key 的 buffer registrations 和 file registrations。
5. 在专用 worker 或当前同步上下文中逐个执行 direct ops；HyFile 内部仍全进程串行。
6. direct key 全部 op 成功后标记为 direct-complete 候选。
7. direct key 任一 op 失败：
   - `kUntouched` 或 `kStoppedSafeToOverwrite` 且策略允许：把整 key 加入 fallback；
   - `kUnknownMayStillBeWritten`：fail-stop；
   - contract violation：请求失败，不掩盖为普通 fallback。
8. direct 阶段结束后，只为 initial fallback 和 safe failed-direct keys 分配 host arena。
9. 调用现有 `Client::BatchGet` 读取完整对象。
10. 为 session-cache 和 fallback key 构建并提交 `DfsAsyncScatterContext`。
11. 同步所有已使用 H2D streams。
12. 对所有成功候选执行 session generation/lease 发布检查。
13. `RealClient` 通过每个计划项保存的 `original_idx` 回填原始结果数组；成功值为该原始 key 的 `sum(all_sizes[original_idx])`，失败值为稳定负错误码。
14. 释放 session pin、file pin、buffer pin、source handle 和 arena。

首版不追求 xDS 与 H2D 并行；先建立可证明的顺序和故障语义。后续只有在测试覆盖完成后才允许并行不同 key 的 direct 与 fallback。

#### 5.2.3 session cache

- 有效完整对象 cache 命中优先于 xDS，继续走 H2D；
- cache source `shared_ptr<BufferHandle>` 持有到 H2D stream synchronize 完成；
- xDS range 成功不创建伪造 host handle，不填充完整对象 cache；
- fallback 完整对象成功后，可以按现有策略进入 session cache；
- session generation 不匹配时不得插入 cache；
- `batch_get_session_end` 删除 map entry，但不能提前释放被在途 H2D 持有的 source。

#### 5.2.4 返回语义

一个 key 只有在以下条件全部满足时返回请求字节数：

```text
该 key 的全部 ranges 已完成
direct synchronize 或 H2D synchronize 成功
没有 range 失败
session generation 仍有效
lease 未过期
目标 registration generation 仍有效
```

混合 batch 中某个 key 的失败不自动覆盖其他 key 的结果，但 backend fatal 会触发 worker fail-stop。

返回与索引映射必须满足：

- `AcceleratorFileIo::Read` 返回单个物理 op 的实际字节数，只用于 exact-length 校验；
- `DistributedStorageBackend::BatchReadRanges` 和 `Client::BatchReadDfsRanges` 返回与输入等长、同序的逐 key `expected<void, XdsError>`；
- backend 内部 coalescing 使用局部 `request_index` 把每个 op 的结果聚合回 key，不接触公开 `original_idx`；
- `RealClient` 的 direct、initial fallback、safe failed-direct fallback 子计划均保存 `request_index -> original_idx` 映射，最终只有 `RealClient` 计算业务成功字节数；
- 任一子层返回数量不匹配时，该子批次全部以 `INTERNAL_ERROR` 失败；
- 每 key 请求字节总和超过 `INT_MAX` 时在提交任何 I/O 前返回 `INVALID_PARAMS`。

### 5.3 对齐和 GPU bounce

#### 5.3.1 首个版本

首个版本只支持完全对齐 direct range：

```text
file_offset % 4096 == 0
uintptr(dst) % 4096 == 0
length % 4096 == 0
```

不对齐 key 走 host fallback。

#### 5.3.2 后续 GPU bounce

Phase 3 可以增加每 device 预分配、预注册的 GPU bounce pool：

```text
对齐覆盖窗口 DFS range
    ↓ xDS
registered GPU bounce slot
    ↓ D2D
最终 DCU KV range
```

必须保证：

- 覆盖窗口完全位于当前对象合法 `[value_offset, value_offset + aligned_size)`；
- 不读取相邻 key 或相邻租户数据；
- bounce slot pin 覆盖 xDS read、synchronize 和 D2D completion；
- D2D 失败时 key 不发布成功；
- GPU bounce 不计入 host staging/cache 容量；
- BUCKET 尾部 padding 可作为合法覆盖窗口，但不能写越最终目标 buffer。

GPU bounce 不属于首个可合并版本。

## 6. 错误处理、回退与运行时恢复

### 6.1 错误、fallback 和 circuit breaker

#### 6.1.1 阶段化处理

| 失败阶段 | 目标状态 | 首版行为 |
|---|---|---|
| 参数、range、registration eligibility | untouched | host fallback 或参数错误 |
| `dlopen`/`dlsym` | untouched | auto 禁用；required 启动失败 |
| driver open/preflight | untouched | auto 禁用；required 启动失败 |
| direct fd open/file register | untouched | 策略允许时该 key fallback |
| buffer register | untouched | registration 不发布，启动/注册失败 |
| `hipSetDevice` before read | untouched | contract/fatal，禁止调用 read |
| `hipFileRead` 短读或错误，sync 成功 | stopped-safe | 整 key fallback |
| `hipFileRead` 后 sync 失败 | unknown | backend fatal，worker fail-stop |
| 成功 read 后 sync 失败 | unknown | backend fatal，worker fail-stop |
| H2D fallback 失败 | stopped | key 失败，不发布 |
| session/lease 发布检查失败 | stopped | 返回 session/lease 错误，不发布 |

#### 6.1.2 不允许 fallback 的错误

- descriptor/path 校验失败；
- object range 越界；
- batch 目标地址重叠；
- session 不存在或 generation 不匹配；
- lease expired；
- `FILE_NOT_FOUND`；
- internal contract violation；
- `kUnknownMayStillBeWritten`；
- required 模式下禁止 request fallback 的配置。

#### 6.1.3 整 key fallback

direct key 任何 op 失败且可安全 fallback 时：

- 丢弃此前 direct op 的结果；
- 记录 discarded direct bytes；
- 延迟申请完整对象 host arena；
- 重新读取完整对象；
- H2D 覆盖该 key 的全部请求 ranges；
- 不只补写失败 range。

#### 6.1.4 Circuit breaker

状态：

```text
CLOSED → OPEN → HALF_OPEN → CLOSED
```

规则：

- `kNotEligible` 不计 backend 失败；
- contract violation 计独立 bug metric，不用普通重试掩盖；
- safe transient I/O 失败计 breaker failure；
- 达到阈值后 OPEN，后续 key 直接 fallback；
- 冷却后只允许一个健康 probe 进入 HALF_OPEN；
- probe 成功恢复 CLOSED；
- backend fatal 不进入普通 cooldown 恢复，直接 fail-stop。

Circuit breaker 不能取消已经卡住的 `hipFileRead`。

### 6.2 Hung I/O 和进程恢复

首版 vendor ABI 没有经过验证的 request timeout/cancel。进入 `hipFileRead` 后不能从另一个线程强制 deregister buffer 或 file。

生产部署必须提供外部 watchdog：

1. worker 定期上报 heartbeat 和当前 xDS op 开始时间；
2. 超过部署配置的 hung-I/O deadline 后，supervisor 终止整个 worker 进程；
3. supervisor 启动新 worker，重新初始化 driver、file handle 和 buffer registration；
4. 不尝试在原进程中恢复未知 DMA 状态。

deadline 根据真实 DFS 尾延迟确定，不能把普通慢 I/O 误判为 hung。该阈值属于部署配置，不由 `AcceleratorFileIo` 自行中断系统调用。

### 6.3 Fork 和多 device

- driver、file handle、buffer registration 都是进程资源；
- 每个 owner 和 pin 保存创建 PID；
- 检测到 PID 改变时，不调用继承资源的 vendor deregister；
- 子进程把继承对象标记为 abandoned，清空可查询 registry，并拒绝继续使用继承的 HyFile runtime；
- 由 fresh exec 启动的新 worker 从 `hipFileDriverOpen`、preflight、buffer registration 和 file registration 完整重建；
- 每个 device pool 分别注册并保存 device id；
- 每次 register/read/synchronize/deregister 前设置正确 device context；
- 首版同一 key 的 ranges 必须位于同一 device；一个 batch 可以包含不同 device 的不同 key，但 HyFile API 仍全进程串行。

## 7. 初始化、配置与写路径边界

### 7.1 启动 preflight

#### 7.1.1 配置和权限

生产必须配置位于真实目标 Parastor/HYFS mount 上的可写 preflight 目录：

```text
MOONCAKE_XDS_PREFLIGHT_DIR
```

目录不可用时：

- `xds-auto`：禁用 xDS 并记录结构化原因；
- `xds-required`：启动失败。

#### 7.1.2 每 device 已知数据读回

对每个目标 device：

1. 分配一个 4096 对齐、长度 4096 的 device probe buffer；
2. 分配一个同样对齐的 host probe buffer并填充 `0xA5`；
3. `hipSetDevice(device_id)`；
4. `hipFileBufRegister`；
5. 在 preflight 目录以随机唯一名和 `0600` 创建文件：

   ```text
   O_RDWR | O_CREAT | O_EXCL | O_DIRECT | O_CLOEXEC
   ```

6. 用 POSIX `pwrite` 写入 host probe buffer，要求 exact-length，并完成必要的文件同步；
7. 注册 opaque fd handle；
8. 清零 device probe buffer；
9. `hipFileRead` 读回，要求 exact-length；
10. `hipDeviceSynchronize`；
11. D2H 读取 device probe buffer并与 host probe buffer 逐字节比较；
12. 逆序 deregister/close/delete。

任一步失败都使该 device 不可用，但不自动覆盖其他 device 的结果。每个 device 分别记录 `kAvailable` 或带结构化原因的 `kUnavailable`。

清理失败单独记录 cleanup metric，并使 preflight 失败。日志可以记录 probe 文件名和错误码，不记录 probe 数据。

#### 7.1.3 多 device 发布和降级

目标 device 集合来自显式程序配置或 SGLang KV pool 初始化时确定的实际 device 集合。若 backend 初始化时尚不能确定全部目标 device，则某个新 device 的第一个 KV pool buffer 注册之前必须同步完成该 device 的 preflight；在 preflight 完成前不得把其 registration 发布为 xDS eligible。

`xds-auto` 采用逐 device 降级：

- 公共的 `dlopen`、required symbols、driver open 或 mount 检查失败时，整个 xDS backend 禁用并继续使用 POSIX；
- 单个 device preflight 失败时，仅将该 device 标为 `kUnavailable`，其他 `kAvailable` device 继续使用 xDS；
- 失败 device 上的 buffer 按 3.3.2 发布为 TE-only、xDS-ineligible；
- 请求按 key 所属 device 独立分类，该 device 不可用时整个 key 走 host fallback；
- 同一个 key 的全部 ranges 仍必须属于同一 device，一个 batch 可以包含不同 device 的不同 key。

`xds-required` 对目标 device 集合采用全有或全无：

- 初始化时已知的任一目标 device preflight 失败，则 backend 初始化失败；
- 初始化后首次出现的新目标 device 必须先通过同步 preflight，否则该 device 的 `register_buffer` 失败；
- 已发布 device 在运行期进入 `kFatal` 时，不静默降级为 auto，backend/worker 进入 unhealthy 或 fail-stop；
- 普通请求因 range 对齐或长度条件不适用时是否允许 host fallback，仍由 `MOONCAKE_XDS_ALLOW_REQUEST_FALLBACK` 控制，这与 preflight 的全有或全无语义分开。

因此 backend 只保留动态库、driver 和 mount 的全局基础能力；实际 direct eligibility 必须同时查询对应 `device_id` 的 `DeviceXdsState`，不能依赖一个全局 `available` 布尔值。

### 7.2 业务写路径

#### 7.2.1 首版

| 操作 | 行为 |
|---|---|
| DFS → DCU KV | 实现 xDS direct read |
| DCU KV → DFS | 保持现有 D2H staging + `BatchWrite` |
| `hipFileWrite` | 首版不加载、不调用 |
| 生产 direct write | 不支持 |

业务 `DistributedStorageBackend::BatchWrite` 不得调用 `AcceleratorFileIo::Write`；首版接口甚至不暴露业务 `Write` 方法。

#### 7.2.2 后续评估

未来业务 xDS write 必须另行设计：

- partial write；
- checksum；
- durability 和 fsync；
- 写完成后再向 Master publish；
- revoke/rollback；
- BUCKET padding；
- 后台写期间源 KV block pin；
- crash consistency；
- write/read 并发和 driver lock。

不得仅把 `hipFileRead` 替换为 `hipFileWrite` 就进入生产。

### 7.3 配置

配置优先级：显式程序配置 > 环境变量 > 默认值。

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `MOONCAKE_DFS_DIRECT_IO_BACKEND` | `posix` | `posix`、`xds-auto`、`xds-required` |
| `MOONCAKE_XDS_LIBRARY_PATH` | `/opt/hyhal/lib/libhyfile.so` | 显式动态库路径 |
| `MOONCAKE_XDS_PREFLIGHT_DIR` | 空 | 真实 mount 上的可写 probe 目录；启用 xDS 时必需 |
| `MOONCAKE_XDS_ALLOW_REQUEST_FALLBACK` | `true` | safe request failure 是否允许整 key fallback |
| `MOONCAKE_XDS_MIN_READ_SIZE` | `65536` | 每个 coalesced DirectReadOp 的最小长度 |
| `MOONCAKE_XDS_FILE_CACHE_SIZE` | `1024` | BUCKET xDS file owner LRU 上限 |
| `MOONCAKE_XDS_BREAKER_FAILURE_THRESHOLD` | `3` | safe transient failure 打开 breaker 的阈值 |
| `MOONCAKE_XDS_BREAKER_COOLDOWN_MS` | `30000` | HALF_OPEN 前冷却时间 |
| `MOONCAKE_XDS_PLANNER_DRY_RUN` | `false` | 只分类和记录，不执行 xDS |
| `MOONCAKE_XDS_GPU_BOUNCE_BYTES` | `0` | 首版必须为 0；Phase 3 才支持非零 |

首版不提供：

```text
MOONCAKE_XDS_ALIGNMENT
MOONCAKE_XDS_SERIALIZE_DMA
MOONCAKE_XDS_EXPERIMENTAL_BATCH
```

对齐和串行是首版安全契约，不是调优开关。

模式语义：

```text
posix:
    不加载 libhyfile.so，不注册 xDS buffer

xds-auto:
    初始化/preflight 失败 → 使用 POSIX
    普通不适用请求 → POSIX fallback
    safe transient request failure → 策略允许时 fallback
    backend fatal → worker fail-stop

xds-required:
    初始化/preflight 失败 → 启动失败
    不适用请求 → 只有 ALLOW_REQUEST_FALLBACK=true 才允许 fallback
    backend fatal → worker fail-stop / 服务 unhealthy
```

启动日志输出最终配置、capabilities、部署版本信息、对齐和 preflight 结果，不输出业务数据。

## 8. 可观测性与测试

### 8.1 可观测性

metrics 标签只能使用有限枚举，不得把 key、路径、错误文本作为常驻 label。

至少增加：

```text
mooncake_dfs_read_requests_total{backend,allocator,result}
mooncake_dfs_read_bytes_total{backend,allocator}
mooncake_dfs_read_latency_seconds{backend,allocator}

mooncake_xds_planned_keys_total{classification,reason}
mooncake_xds_planned_ranges_total{classification,reason}
mooncake_xds_planned_bytes_total{classification,reason}
mooncake_xds_coalesced_ops_total

mooncake_xds_read_calls_total{result,error_class}
mooncake_xds_read_bytes_total
mooncake_xds_discarded_direct_bytes_total{reason}
mooncake_xds_dma_seconds
mooncake_xds_device_sync_seconds{result}
mooncake_xds_driver_lock_wait_seconds
mooncake_xds_pending_ops

mooncake_xds_fallback_keys_total{reason}
mooncake_xds_fallback_ranges_total{reason}
mooncake_xds_fallback_bytes_total{reason}

mooncake_xds_registered_device_bytes{device_id}
mooncake_xds_registered_buffers{device_id,state}
mooncake_xds_buffer_registration_mode_total{device_id,mode,reason}
mooncake_xds_registered_files{allocator}
mooncake_xds_registration_failures_total{operation}
mooncake_xds_cleanup_failures_total{operation}

mooncake_xds_circuit_breaker_state
mooncake_xds_backend_fatal_total{operation}
mooncake_xds_preflight_total{device_id,result,stage}
mooncake_xds_device_state{device_id,state}
```

一次 batch trace 建议包含：

```text
trace_id
keys
ranges
requested_bytes
eligible_bytes
direct_bytes
discarded_direct_bytes
fallback_bytes
coalesced_ops
driver_lock_wait_us
dma_us
device_sync_us
h2d_us
total_us
first_error_class
first_error_operation
```

不记录完整 key 或敏感业务数据。需要 key 关联时只使用当前 trace 范围内的不可逆短 hash。

端到端继续保留 SGLang prefetch latency、TTFT、ITL、CPU 使用率、host memory bandwidth 和 DCU copy-engine 利用率。

### 8.2 测试方案

#### 8.2.1 Planner 单元测试

- SHARD 和当前 BUCKET `file_offset` 计算；
- 空 ranges、零长度、null dst；
- object offset/length/file offset/address 的整数溢出；
- object range 越界；
- batch 重复 key；
- 同 key和跨 key目标地址重叠；
- 未注册、跨 registration、draining registration；
- PID/generation mismatch；
- mixed-device key；
- 每 key 请求字节总和超过 `INT_MAX`；
- file/device/length alignment；
- `MIN_READ_SIZE`；
- 只有 file 和 device 同时连续才 coalesce；
- coalesce 不跨对象、不越 registration、不超过 `SSIZE_MAX`；
- 每 key 全 direct 或全 fallback。

#### 8.2.2 Registry 单元测试

- 相同 base/size/device 重复注册引用计数；
- 相同 base 不同 size/device 拒绝；
- 部分重叠拒绝；
- 注册各阶段失败时逆序回滚；
- `posix` 发布 TE-only entry；
- `xds-auto` 中安全的 xDS 注册失败发布 TE-only entry 且公开注册成功；
- `xds-required` 中 xDS 注册失败回滚 TE registration 且公开注册失败；
- xDS 注册或回滚状态未知时不得发布 TE-only entry，并进入 quarantined/fail-stop；
- unsupported memory kind 和 preflight 失败 device 在 auto 模式下标记 xDS-ineligible；
- pin 与 unregister 并发；
- draining 后拒绝新 pin；
- inflight 归零后才 deregister；
- deregister 失败进入 quarantined/fail-stop；
- fork/PID 改变后不调用继承 owner 的 vendor cleanup；
- 多 device 分别注册。

#### 8.2.3 Fake backend 单元测试

- 捕获 `hipFileRead` 五个语义参数；
- 原始 registered base 与 pool-relative device offset；
- exact-length success；
- 正短读；
- 当前手工 ABI 已定义及未知的 vendor 错误；
- read error 后 synchronize success → safe fallback；
- read error 后 synchronize failure → fatal；
- success read 后 synchronize failure → fatal；
- 任意时刻最多一个 HyFile API 临界区；
- DeviceGuard 选择正确 device；
- owner/pin 生命周期；
- file cache eviction 与 fd reuse；
- backend 返回结果与输入等长同序，内部 coalescing 后仍能按局部 request index 聚合；
- 返回数量不匹配时子批次全部得到 `INTERNAL_ERROR`；
- `RealClient` 使用 `original_idx` 回填混合 direct/fallback batch，并且只有它计算公开成功字节数；
- circuit breaker 状态转换。

#### 8.2.4 Session 和路由测试

- session cache hit 优先；
- cache miss direct；
- initial ineligible fallback；
- safe failed-direct 整 key fallback；
- direct 不分配 host arena、不提交 H2D、不写完整对象 cache；
- fallback 成功后按现有配置写 session cache；
- session end 与 direct I/O 并发；
- 同 key 新 session generation 不接受旧完成；
- lease 在 I/O 中过期时不发布成功；
- mixed batch 等待全部 direct/H2D completion 后返回；
- backend fatal 触发 fail-stop hook。

#### 8.2.5 真实 DTK/xDS 集成测试

在真实海光节点和 Parastor/HYFS mount 上覆盖：

1. 每 device 启动 round-trip preflight；
2. `xds-auto` 中一个 device preflight 失败时仅该 device fallback，其他 device 保持 direct；
3. `xds-required` 中任一目标 device preflight 失败时初始化或对应首次注册失败；
4. 初始化后首次出现的新 device 在 buffer 发布前完成同步 preflight；
5. 单文件、单 device、单 aligned range；
6. 单文件、多 range；
7. 多文件；
8. 多 device batch；
9. SHARD；
10. BUCKET fully-aligned range；
11. 不对齐 file/device/length 的 fallback；
12. 不支持的 mount；
13. `libhyfile.so` 缺失和 required symbol 缺失；
14. runtime/driver 版本不匹配；
15. direct fd 必须包含 `O_DIRECT`；
16. fallback 使用独立 fd；
17. fork 后完整重建；
18. handle/buffer 重复注册和解除注册；
19. read 短返回和错误注入；
20. synchronize 失败触发 worker 退出；
21. hung-I/O watchdog 重启；
22. 多线程压力下仍只有一个 HyFile 临界区；
23. 24 小时 soak 检查 fd、registration、显存和 driver 资源。

所有读取与 POSIX 基准逐字节比较，不能只验证返回长度。

#### 8.2.6 禁止 host 中转验证

在 direct 测试中 hook/观测：

```text
host arena allocation
pread/preadv/readv/mmap payload read
H2D copy
```

DIRECT range 出现上述任意行为都不能计为 direct success。允许控制面的小对象、日志和元数据分配，但必须与 payload host staging 区分。

#### 8.2.7 SGLang E2E

- layer-wise load；
- page-wise load；
- ReadPlan；
- MHA、MLA、Mamba 等实际 pool layout；
- tensor/data/pipeline parallel；
- 连续 prefill/decode；
- session end、取消、lease 到期；
- cache eviction；
- 开启/关闭 session cache；
- 同模型、prompt、KV layout 和请求序列对比输出；
- load 失败后目标 block 不被消费；
- xDS disabled 时行为和性能基线不显著回退；
- 现有 DFS 写路径保持不变。

#### 8.2.8 性能矩阵

| 维度 | 建议取值 |
|---|---|
| 单 coalesced op | 4 KiB、64 KiB、256 KiB、1 MiB、4 MiB、16 MiB |
| batch keys | 1、8、32、128 |
| ranges/key | 1、2、8、32 |
| planner/调用线程 | 1、8、32、128 |
| allocator | SHARD、BUCKET |
| path | POSIX buffered、O_DIRECT host、xDS direct、后续 xDS bounce |
| device count | 1、2、4、8 |
| SGLang mode | layer-wise、page-wise、ReadPlan |

特别记录每个 `hipDeviceSynchronize` 对同设备无关 compute 的影响。

## 9. 实施计划、代码改动与构建

### 9.1 分阶段实施计划

Phase 0～2 可以作为一个开发任务、一个分支和一个生产候选版本一次性交付，但必须保持独立模块、逻辑提交和验收门禁，不得将 PoC、生命周期基础设施和 `RealClient` 热路径写成不可拆分的一体化实现。推荐至少拆分为以下逻辑提交；是否拆成多个 PR 由项目协作方式决定：

```text
Commit 1: reusable HyFile ABI declaration/backend + thin standalone PoC
Commit 2: planner/registry/file owner/session lifecycle + fake tests
Commit 3: BUCKET RealClient integration + E2E/metrics/fallback
```

一次性交付不表示可以跳过中间验证。验收顺序固定为：

```text
Phase 0 → Phase 0.5 → Phase 1 → Phase 2
```

后续阶段代码可以提前存在于同一分支，但前一阶段退出条件未满足时，不能启用后一阶段的数据面；默认 backend 始终为 `posix`。Phase 2 是 BUCKET 生产场景首个真正打通 xDS read 的阶段。Phase 2.5 的 SHARD 支持不阻塞 BUCKET MVP，可以独立后置。

#### Phase 0：ABI 和独立 PoC

- 固化并评审 `hyfile_abi.h` 的最小手工 ABI 声明和版本矩阵；
- 确认 `libhyfile.so` 路径和 required symbols，记录版本/build-id 但不做硬编码拒绝；
- 确认 `hipFileRead` 精确原型和错误语义；
- 在生产目录中实现可复用的 `hyfile_abi.h`、最小 `HyFileBackend` 和 RAII file/buffer handles；
- PoC 仅提供独立的薄 `main()`，复用上述生产模块，不维护第二套 vendor 调用实现；
- 独立程序验证 O_DIRECT opaque fd；
- 每 device 整 pool/subrange 注册语义；
- POSIX 写入已知数据，xDS exact-length read，并逐字节校验；
- 4096 对齐；
- 进程级串行压力；
- fork 子进程拒绝继承的 vendor 资源，由 fresh exec 的新 worker 重建；
- hung-I/O 和 synchronize 语义。

退出条件：在目标版本矩阵上稳定完成逐字节正确的已知数据读回，错误注入语义明确，且没有 Mooncake CPU fallback。

#### Phase 0.5：Planner dry-run

- 以 BUCKET 为主要 allocator，在当前生产型 SGLang workload 中运行 planner；
- 不调用 HyFile；
- 收集 allocator 分布、eligible bytes、op count、alignment failure 和 read amplification；
- 按模型、dtype、page size 和并行配置形成报告。

退出条件：典型 BUCKET workload 有足够 direct eligible bytes，预计调用/同步开销不会明显抵消读放大收益。若覆盖率不足，Phase 2 仍可用于功能打通和受限灰度，但不得把低覆盖率包装成生产性能收益。

#### Phase 1：内部抽象和生命周期

- `AcceleratorFileIo`；
- fake backend；
- owner/pin 类型；
- composite buffer registry；
- BUCKET canonical path、file identity、动态 file owner/LRU 和 xDS registration cache；
- BUCKET eviction/recovery/fd reuse 与 inflight read 的生命周期协调；
- session generation/pin；
- batch target-overlap validation；
- planner；
- metrics/circuit breaker；
- SGLang 显式 unregister 顺序。

退出条件：所有单元测试和故障路径通过，不接真实热路径。

#### Phase 2：BUCKET fully-aligned read-only MVP

- 固化并启用 Phase 0/1 的手工 ABI 声明和 `HyFileBackend`；
- per-device preflight；
- 新增 `BatchReadDfsRanges`；
- 接入 `batch_get_into_multi_buffer_ranges`；
- 只支持 BUCKET、完全对齐、达到阈值的 key；
- 复用当前对齐的 `BucketEntryLayout`，不改变磁盘格式；
- bucket id → canonical path 重建和 descriptor/range 校验；
- BUCKET file owner/LRU、xDS file registration cache 和 inflight pin；
- bucket eviction 时停止新 pin、等待 inflight drain，再 deregister/close；
- 验证 restart recovery、cache eviction 和 fd reuse；
- 同步 read、global mutex、exact-length、per-op device synchronize；
- safe failure 整 key fallback；
- unknown target state fail-stop。

退出条件：真实 BUCKET 硬件 E2E correctness、fallback、fail-stop、零 host payload staging、cache eviction、fd reuse、灰度开关和 soak 全部通过。达到此条件即视为 BUCKET xDS read 路径真正可用，可开始受控灰度。

#### Phase 2.5：SHARD fully-aligned read-only compatibility（可选后置）

- 为固定、预分配的 shard 文件建立长生命周期 file owner 和 xDS registration；
- 校验 `shard_idx`、descriptor path 与配置的 shard path 完全一致；
- 复用 Phase 2 的 planner、buffer/session pin、执行器、fallback 和 fail-stop 语义；
- 不改变 SHARD 磁盘格式或 allocator 协议。

退出条件：SHARD 真实硬件 E2E correctness、文件生命周期、fallback 和 soak 通过。生产无 SHARD 需求时，本阶段不阻塞 BUCKET 版本发布。

#### Phase 3：GPU bounce

- 每 device 对齐 GPU bounce pool；
- 不对齐覆盖窗口验证；
- xDS → bounce → D2D；
- slot pin 和 D2D completion；
- 根据 Phase 0.5/2 metrics 决定是否启用。

#### Phase 4：优化和独立写设计

- 根据厂商正式语义评估减少 synchronize 次数；
- 评估异步/batch API；
- 独立设计业务 xDS write、durability 和 publish 顺序；
- 不扩展未经确认的 vendor ABI。

### 9.2 代码改动清单

| 模块 | 改动 |
|---|---|
| SGLang `mooncake_direct_linker.py` | 跟踪成功注册的 storage allocation；shutdown 显式 unregister；保持 pool owner 存活 |
| `real_client.cpp/.h` | session generation/pin；cache/direct/fallback 路由；batch 目标冲突校验；registration pin；发布检查 |
| `client_service.cpp/.h` | `BatchReadDfsRanges`；结果和 metrics 合并；保持现有业务写路径 |
| `distributed_storage_backend.cpp/.h` | descriptor/value offset 解析；planner；SHARD/BUCKET direct file owner；串行 direct 执行 |
| `fs_adapter.*` | 保持普通 I/O；仅补充安全获得 canonical identity/fstat 信息的接口 |
| `device/*` | 注册阶段解析 device metadata；DeviceGuard；sync/fail-stop hook |
| `registered_pinned_memory.*` 或新 registry | composite TE+xDS registration、generation、pin、drain、quarantine |
| `storage/distributed/xds/*` | interface、HyFile backend、手工 ABI 声明、file cache、breaker、preflight |
| CMake/wheel | `USE_XDS` 可选构建；Linux/HIP 约束；不要求 SDK header；不打包 vendor runtime |
| metrics | planner/direct/fallback/registration/lock/sync/fatal 指标 |
| tests | planner、registry、fake backend、真实 DTK、SGLang E2E、性能和 soak |

建议文件结构：

```text
mooncake-store/include/storage/distributed/accelerator_file_io.h
mooncake-store/include/storage/distributed/xds/hyfile_backend.h
mooncake-store/include/storage/distributed/xds/xds_buffer_registry.h
mooncake-store/include/storage/distributed/xds/xds_read_planner.h

mooncake-store/src/storage/distributed/xds/hyfile_backend.cpp
mooncake-store/include/storage/distributed/xds/hyfile_abi.h
mooncake-store/src/storage/distributed/xds/xds_buffer_registry.cpp
mooncake-store/src/storage/distributed/xds/xds_read_planner.cpp

mooncake-store/tests/xds_read_planner_test.cpp
mooncake-store/tests/xds_buffer_registry_test.cpp
mooncake-store/tests/xds_backend_fake_test.cpp
```

`hyfile_abi.h` 是唯一的 vendor ABI 声明源，`hyfile_backend.cpp` 是唯一执行
`dlopen`/`dlsym` 和调用 vendor 函数的编译单元。

### 9.3 构建

建议构建参数：

```text
-DUSE_XDS=ON
```

规则：

- 普通构建默认 `USE_XDS=OFF`；
- `USE_XDS=ON` 要求 Linux、HIP 构建环境，但不要求 SDK header；
- 始终使用仓库内唯一的最小手工 ABI 声明，并执行静态布局校验；
- 动态库运行时加载；
- fake backend CI 不依赖海光硬件；
- 真实 DTK 测试使用单独硬件 CI/验收环境。

## 10. 验收、风险与待确认项

### 10.1 验收标准

#### 10.1.1 正确性和安全

- planner、registry、session、fallback 和故障测试全部通过；
- 所有真实读取与 POSIX 结果逐字节一致；
- direct path payload host staging bytes 为 0；
- direct path H2D payload bytes 为 0；
- 短读从不被视为成功；
- safe failure 只执行整 key fallback；
- unknown target state 必定 fail-stop；
- session generation/lease 失效时不发布成功；
- target overlap 在任何 I/O 前被拒绝；
- fork 后不访问继承的 vendor 资源；
- 24 小时 soak 无 fd、registration、显存或 driver 资源增长。

#### 10.1.2 兼容性

- `MOONCAKE_DFS_DIRECT_IO_BACKEND=posix` 时行为与当前版本一致；
- xDS 构建关闭时不引入 vendor runtime 依赖；
- SGLang 公开 API不变；
- Master 元数据协议不变；
- 当前业务 DFS 写路径不变；
- auto 模式缺少库、符号或不支持 mount 时可安全回退。

#### 10.1.3 性能

平台和模型的数值门槛在 Phase 0.5 后固化，但至少必须满足：

- 典型大块 aligned L3 hit 的端到端带宽稳定优于当前路径；
- TTFT 的 p50/p95/p99 不因 device-wide synchronize 出现不可接受回退；
- CPU 利用率和 host memory bandwidth 明显下降；
- pinned host staging 峰值下降；
- xDS disabled 不产生显著性能回退；
- 小 range/低 eligibility workload 不因误走 xDS 而退化，能够由 planner 阈值回退。

只有 `hipFileRead` 带宽提升而 SGLang TTFT 无提升，不视为验收通过。

### 10.2 风险与缓解

| 风险 | 缓解措施 |
|---|---|
| `libhyfile.so` ABI 随 DTK 变化 | 单一手工 ABI 声明、最小 required symbols、版本清单、真实 mount preflight；不兼容时 fail closed |
| HyFile/driver 对目标 DFS 不支持或在内部静默 bounce | mount 白名单、preflight、direct/fallback 指标；没有厂商 counter 时不宣称驱动内部零 bounce |
| DCU buffer 生命周期错误 | eager composite registration、generation、owner/pin、unregister drain 和 quarantined 状态 |
| BUCKET 动态文件被淘汰、替换或 FD 被复用 | canonical identity、file owner/inflight pin、有界 LRU、淘汰 drain、重新注册 |
| 大量小 range 的调用与同步开销超过收益 | `MIN_READ_SIZE`、双向连续 coalescing、Phase 0.5 telemetry 和性能门槛 |
| HyFile 非线程安全 | 首版使用进程级 mutex 串行全部 vendor API，实机并发压力测试 |
| 部分完成或未知 DMA 状态使错误 KV 可见 | exact-length、device synchronize、每 key 聚合结果；未知 target state 必须 fail-stop |
| 同步 I/O 永久阻塞 worker | circuit breaker 阻止后续调用，外部 watchdog/supervisor 重启进程 |
| 注册显存或文件句柄资源过高 | 整 pool 注册、注册字节数指标、有界 file cache、启动 fail-fast 和 soak |
| direct path 绕过 descriptor、路径或 lease 校验 | 强制复用 canonical resolver，提交前和发布前分别校验 session generation/lease |

### 10.3 实现前仍需厂商确认

1. 厂商对 soname/路径、calling convention 和 ABI 演进规则的正式说明。
2. 首版 `hipFileRead` 的精确参数类型、返回类型和错误传递方式；后续若设计 write，再单独确认 `hipFileWrite`。
3. Parastor/HYFS 支持的 client、服务端、mount 模式和版本范围。
4. FUSE 是否明确不支持。
5. 4096 对齐是硬要求还是当前版本限制。
6. 不对齐请求是报错、短读还是内部 bounce。
7. 同步 read 返回后 DMA 是否已完成；为什么仍需要或不需要 device synchronize。
8. synchronize 失败后的厂商恢复建议。
9. 多 device、NUMA、NIC 和 Parastor path 的亲和规则。
10. buffer 数量、注册显存总量和 pinned 资源上限。
11. 是否存在正式 request timeout/cancel。
12. 哪些错误允许重试，哪些必须重建 driver/context/进程。
13. 是否有 direct-DMA bytes、CPU-bounce bytes 或队列 trace。
14. file/buffer register/deregister 与 read/write 是否全部要求串行。

未绑定实际函数的 IO params、event、poll 或 batch 结构一律视为影子 ABI，首版不得使用。上述待确认项按照默认的定。

### 10.4 首个可合并版本的最终契约

第一个生产候选版本严格限定为：

- 业务 read-only xDS；
- BUCKET allocator；
- 仅真实 preflight 成功的 Parastor/HYFS mount；
- `libhyfile.so` 显式 `dlopen`；
- `O_DIRECT` opaque fd；
- SGLang KV pool eager registration；
- 注册阶段解析 device id；
- 完全对齐且达到阈值的 ranges；
- bucket id → canonical path 重建和 descriptor/range 校验；
- BUCKET file owner/LRU、xDS registration cache 和 inflight pin；
- bucket eviction、restart recovery 和 fd reuse 生命周期安全；
- batch key 唯一且目标地址不重叠；
- 每 key 全 direct 或全 fallback；
- 同步 `hipFileRead`；
- exact-length；
- 全进程 HyFile API mutex；
- 每 op 返回前 device synchronize；
- safe failure 整 key fallback；
- unknown target state worker fail-stop；
- session generation 和完成前 lease 校验；
- 不实现 GPU bounce；
- 不实现业务 xDS write；
- 不实现 async/batch/event/poll；
- 不实现进程内 timeout/cancel；
- `posix` 默认，灰度时显式选择 `xds-auto`；
- `xds-required` 用于验收和强制部署；
- 一键切回当前 POSIX/O_DIRECT + host staging 路径。

该范围可以验证核心价值：在不改变 SGLang 对外接口和 Master 协议的前提下，消除 fully-aligned DFS KV range 的完整对象 host staging 和 H2D scatter，同时对不适用和可恢复错误保留现有稳定路径。

### 10.5 参考资料

- 当前代码基线：`mooncake-das` 的 `feat/optimize_prefetch`，以及 `sglang-das` 的 `feature/optimize_the_direct_linker`/`588405a92a`。
- LMCache xDS/HyFile 适配代码审阅记录（本设计输入，2026-09）。
- [LMCache GDS backend documentation](https://github.com/LMCache/LMCache/blob/dev/docs/source/kv_cache/storage_backends/gds.rst)：用于对照 direct-storage backend 的配置和测试思路；HyFile ABI 由本项目的最小手工声明、部署版本清单和真实 preflight 共同约束。
- [LMCache GDS context tests](https://github.com/LMCache/LMCache/blob/dev/tests/v1/gpu_connector/test_gds_context.py)：用于对照 GPU buffer/context 生命周期测试。
