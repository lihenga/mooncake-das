# 基于 xDS 的 DFS KV Cache 直接预取设计

## 0. 文档状态与阅读指南

- 状态：实现规格（Implementation-ready Design）
- 目标组件：SGLang Mooncake Direct Linker、Mooncake Store、DistributedStorageBackend
- 首个可合并版本：从 BUCKET allocator 的 DFS 文件读取对齐的 KV Cache range，直接写入海光 DCU 显存
- 业务写路径：首版不实现 xDS 写，继续使用现有 Mooncake DFS 写路径
- xDS 运行时：海光 DTK/xDS、`libhyfile.so`
- DFS 首期范围：在部署前通过独立 round-trip 验收的 Parastor/HYFS mount
- 对齐要求：首版固定 4096 字节，不提供生产覆盖开关
- 当前代码校验基线：
  - `mooncake-das`：commit `dded8f87051d537ba677193cb22941f564f1a330`
  - `sglang-das`：`feature/optimize_the_direct_linker`，commit `c4257ce7b190710b953504645aea1cdd7a53035e`

本文是代码实现依据。实现若需要改变本文标为“首版契约”的行为，必须先更新设计和对应测试，不得通过隐式 fallback 或未记录的兼容逻辑改变语义。

本文以当前首版代码为实现标准，只描述已采用的数据结构、调用路径和必要安全契约。
部署验收与运行时代码分开：真实 mount/DCU round-trip 由独立验收程序执行，不在
Mooncake 启动路径中增加 probe 状态机。本文范围之外的能力必须另行设计和评审，本文
不预设其实现结构。

### 阅读指南

| 阅读目的                               | 重点章节                           |
| ---------------------------------- | ------------------------------ |
| 快速理解做什么、为什么做                       | 1.1～1.4、2.1、10.4               |
| 核对当前代码已经实现到哪里                     | 1.3.6、3.2～3.4、7.3、9.1          |
| 实现 Phase 0～2 的核心代码                 | 2～7、9.1～9.3                    |
| 实现 `RealClient` direct/fallback 路由 | 3.3～6.1                        |
| 实现 `libhyfile.so` shim/backend     | 2.2、3.2、5.1、6、7.1              |
| 实现 BUCKET-first MVP                | 1.3.5、4.2.3～4.2.4、5、6、9.1      |
| 编写测试和验收                            | 8、10.1、10.4                    |

本次 Phase 0～2 直接相关的实现核心是第 3～7 章；第 1～2 章定义背景、范围和不可违反的
契约，第 8～10 章定义观测、测试、实施和验收。本文只规定 BUCKET fully-aligned read-only
路径，其他 allocator、非对齐处理、异步/batch 和业务 xDS write 不在本文范围内。

## 1. 背景、现状与目标

### 1.1 摘要

SGLang 使用 Mooncake 作为 L3 KV Cache。原有 host fallback 路径在只需要对象一部分时，
仍会把完整对象读入 Mooncake pinned host arena，再异步 H2D scatter 到 SGLang 的 DCU
KV pool。当前代码已在这条路径前加入 BUCKET xDS direct 尝试；不适用或可安全降级的
key 仍使用该 host fallback。

本文引入一条 xDS/HyFile 直接读取路径：Mooncake 把 DFS 文件的绝对 offset、已注册 DCU KV pool 的原始基址以及目标 pool-relative offset 传给 `hipFileRead`，使满足条件的 KV range 从 DFS 直接进入最终 DCU 地址。

首版保持以下原则：

1. 不改变 SGLang 公开 KV connector API。
2. 不改变 Master replica 元数据和分配协议。
3. 每个 key 在一次调用中选择全 direct 或全 fallback。
4. xDS direct range 不分配 Mooncake host staging，也不执行 H2D。
5. xDS 不可用或请求不适用时，安全回退现有路径。
6. 某个 direct op 失败时，只有确认 DMA 已停止，才允许整个 key 走 fallback。
7. 当前 Mooncake 没有独立 DFS/xDS helper 进程。若 xDS read 已提交但无法证明 DMA 已停止，
   必须禁止 host fallback，并把 backend 置为 sticky fatal。恢复方式是由外部 supervisor 或
   人工重启整个承载进程；新进程重新初始化 driver、file handle、buffer registration、
   连接和 session。
8. 部署前由独立验收程序通过 POSIX 写入已知数据，再用 `hipFileRead` 读入 DCU 并校验；业务数据面只使用 xDS read。

本文所称“零拷贝”仅表示 Mooncake 不分配 host staging buffer，也不显式执行 H2D payload copy。hipFileRead 返回成功并不能证明 libhyfile.so、DFS client 或内核驱动内部未使用临时 host/device bounce buffer。除非厂商提供可区分 direct-DMA bytes 与 bounce bytes 的可信计数器、trace 或正式语义保证，本文只将该路径称为“Mooncake 零 host staging”，不宣称端到端完全零拷贝。

### 1.2 术语

| 术语                 | 含义                                           |
| ------------------ | -------------------------------------------- |
| xDS                | 海光 DCU 软件栈提供的 GDS 类直接存储能力                    |
| HyFile             | 本设计使用的 xDS 用户态接口，符号以 `hipFile*` 命名           |
| direct range       | 通过 `hipFileRead` 直接写入最终 DCU KV 地址的 range     |
| host fallback      | 当前 DFS → pinned host arena → H2D 路径          |
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

#### 1.3.2 现有 DFS host fallback 路径

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
4. DFS replica 先查询 `get_session_prefetch_cache_`。
5. session cache miss 时，为完整对象分配 aligned pinned host arena view。
6. `Client::BatchGet` 构造现有 `DfsReadRequest`，由 backend 读取完整对象。
7. `DfsAsyncScatterContext` 合并相邻 H2D copy，并同步使用过的 stream。
8. 完成后填写逐 key 字节数并返回。

当前校验基线中不存在 Mooncake `DfsPrefetcher` 或 `TryConsume`。本文不把不存在的
prefetcher 纳入首版路由。上图描述 fallback；实际 DFS cache miss 会先执行 1.3.6 的 xDS
资格筛选，只有未进入 direct 或安全回退的 key 才继续执行步骤 5～8。SGLang 业务层所称
“prefetch”指 L3 KV 恢复行为，不等同于一个名为 `DfsPrefetcher` 的组件。

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

#### 1.3.6 当前 xDS 实现状态

当前代码已经打通以下软件调用链：

```text
SGLang MooncakeDirectLinker
  → RealClient::batch_get_into_multi_buffer_ranges
  → RealClient::execute_session_dfs_range_reads
  → Client::BatchReadDfsRanges
  → DistributedStorageBackend::BatchReadRanges
  → AcceleratorFileIo::Read
  → HyFileAcceleratorFileIo::Read
  → hipFileRead + hipDeviceSynchronize
  → 最终 DCU KV range
```

当前方法归属固定为：`BatchReadDfsRanges` 属于 `Client`，`BatchReadRanges` 属于
`DistributedStorageBackend`。前者只做转发，后者负责文件定位、buffer pin、对齐检查、
coalescing 和逐 op 执行。

当前实现状态如下：

| 能力 | 当前首版行为 |
|---|---|
| HyFile ABI/runtime | 最小手工 ABI；`dlopen`/`dlsym` 7 个 required symbols；进程级 runtime；fork PID 防护 |
| BUCKET direct read | 已接入 RealClient 热路径；完全对齐 range 可直接进入最终 DCU 地址 |
| SHARD direct read | 不支持；`GetXdsFile` 返回 `allocator-not-supported` 并按策略 fallback |
| buffer registry | `std::map<base, shared_ptr<XdsBufferEntry>>`；使用 `logical_refs`、`inflight`、`accepting`、`quarantined` |
| file registry | `xds_files_` 缓存 registration、`shared_ptr<OpenFileHandle>` 和 dev/inode identity；缓存满时返回 `xds-file-cache-full` |
| session 安全 | `get_sessions_` 保存 `QueryResult` 快照；完成后比较 replica/checksum 并复查 lease；同 key 不允许并发 end/start/read |
| fallback | 安全错误整 key fallback；unknown target state 禁止 fallback |
| fatal | backend 设置 sticky `xds_fatal_` 并拒绝当前及后续 xDS 请求 |
| 部署验收 | 目标环境逐 device、真实 mount round-trip 由独立程序执行，不属于 Mooncake runtime 初始化 |

因此当前可以称为“BUCKET read-only xDS 软件路径已打通”。是否生产启用，由第 7.1 节的
独立部署验收和第 10 章的正确性/性能标准决定。

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
- 只返回逐 key 成功或带 `target_state` 的失败，不计算公开 API 的业务成功字节数。

`AcceleratorFileIo::Read` 仍返回底层实际完成字节数，用于强制校验 `actual == requested`；这个字节数是单个物理 direct op 的完成信息，不直接作为 Python API 返回值。`RealClient` 是唯一计算公开逐 key 成功字节数的层：它保留现有 `SessionRangeReadRequest::original_idx`，为 direct/fallback 子计划维护局部 `request_index -> original_idx` 映射，并且仅在该 key 的全部 ranges、设备同步和发布检查均成功后返回 `sum(all_sizes[original_idx])`。`DfsRangeReadRequest` 不携带公开请求下标，backend 不依赖调用方的结果数组布局。

子层返回数量与输入数量不一致时，对应子批次全部返回 `INTERNAL_ERROR`，不得按 key 猜测映射。由于公开返回类型当前为 `int`，planner 还必须在任何 I/O 前验证每 key 请求字节总和不超过 `INT_MAX`。

#### 2.1.2 每 key 单一路径

首版一个 key 中只要有任一 range 不满足 direct 条件，整个 key 走 host fallback。

禁止在一次调用中对同一个 key 同时执行 xDS 和 H2D。xDS 失败后允许 fallback 时，也必须覆盖该 key 的全部 ranges。

#### 2.1.3 `AcceleratorFileIo::Read` 是原子安全边界

一次 `Read` 内部负责：

```text
PID/registration id/边界复核
→ DeviceGuard / hipSetDevice
→ 获取进程级 HyFile mutex
→ hipFileRead
→ hipDeviceSynchronize
→ exact-length 校验
→ 生成稳定错误分类和 target state
→ 释放 HyFile mutex
```

必须先同步、再根据返回字节数分类。原因是短读或 vendor error 之后也要先证明 DMA 已经
停止，才能把目标状态标记为 `kStoppedSafeToOverwrite`。当前
`HyFileAcceleratorFileIo::Read` 已按该顺序实现。

上层不得直接调用 vendor synchronize，也不得直接解释 vendor 返回码。

#### 2.1.4 首版同步和串行

首版：

- 所有 HyFile API 调用由一个进程级 mutex 串行保护；
- `hipFileRead` 和其后的 `hipDeviceSynchronize` 在同一次加锁范围内完成；
- 不提供关闭串行保护的配置；
- 不假设 SGLang compute stream 与 HyFile DMA 自动有序；
- 不使用未确认的 batch/event/poll 结构体或函数。

#### 2.1.5 fatal 状态

如果 `hipDeviceSynchronize` 失败，无法证明驱动不再访问目标显存。首版行为固定为：

```text
标记 backend fatal
→ 拒绝新 xDS 请求
→ 当前请求不 fallback、不发布成功
```

首版不实现 SGLang KV allocator 的 poisoned-block quarantine 协议。

当前代码设置 `xds_fatal_` 并拒绝当前及后续 xDS 请求，不主动终止 worker，也不实现新的
supervisor 协议。部署侧应把 fatal 错误或进程卡死纳入已有健康检查，并通过重启承载进程
恢复；不得在原进程中清除 fatal 后继续复用 vendor 资源。

### 2.2 依赖和兼容性

#### 2.2.1 运行时依赖

- 海光 DCU；
- 海光 DTK/ROCm 用户态运行时；
- 匹配的内核驱动和固件；
- `libhyfile.so`；
- Parastor/HYFS client 和受支持的 mount；
- 支持 `O_DIRECT` 的路径；
- HIP device/context 和 synchronize API；
- 用于进程级恢复的现有外部 supervisor，或明确的人工重启流程。

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

独立部署验收使用 POSIX 写入已知测试数据，再通过 `hipFileRead` 读入 DCU buffer 并
回读校验；业务首版不要求也不加载 `hipFileWrite`。

首版不声明、不加载其他 vendor 符号。

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
5. 启用业务流量前必须用独立验收程序完成真实 mount、逐 device 的已知数据读回；
6. 错误码映射以已确认的当前 ABI 为准，未知错误按保守策略处理。

当前 `hyfile_abi.h` 已按现有参考实现固化并用于编码；以下问题不阻塞软件路径开发，
但仍是目标环境生产验收项：

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

本章是首版实现规范，类型、生命周期和调用关系均以当前代码为准。首版使用 backend
registry 中已有的 `shared_ptr + inflight` 保护 buffer，不再定义另一套抽象资源类型。

### 3.1 总体架构

```text
SGLang Mooncake Direct Linker
  - KV pool lifetime
  - register/unregister buffers
  - batch_get_into_multi_buffer_ranges
                  |
                  v
RealClient
  - session snapshot、replica compatibility 和 lease 复查
  - session cache routing
  - device pointer / same-device / INT_MAX 粗筛
  - per-key direct/fallback classification
                  |
                  v
Client::BatchReadDfsRanges
                  |
                  v
DistributedStorageBackend
  - descriptor validation
  - BUCKET target resolution
  - value/file offset calculation
  - batch target-overlap validation
  - registration interval pinning
  - range coalescing
  - 有界 xDS file registration cache
          |                         |
          v                         v
AcceleratorFileIo              FileSystemAdapter
  HyFile adapter                existing fallback
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

动态库和 driver 初始化结果属于 backend 级状态，device 是否存在和是否已进入 fatal 属于
per-device 状态。`BuildAdapter` 在 `hipGetDeviceCount` 成功后把枚举范围内的 device 初始化为
`kAvailable`；I/O 热路径不重新探测。首版 alignment 来自已验证的部署 ABI，固定为 4096，
不允许普通用户覆盖。真实 DFS mount 与每张目标 DCU 的兼容性由 7.1 的独立部署验收确认，
不在 Mooncake 进程内维护额外 probe 状态机。

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

#### 3.2.3 注册对象与 `DirectReadOp`

当前仓库使用两个轻量 registration value：

```cpp
struct XdsFileRegistration {
    uint64_t id;
    int fd;
};

struct XdsBufferRegistration {
    uint64_t id;
    void* base;
    size_t length;
    int32_t device_id;
};
```

`id` 是 adapter 在当前进程内分配的注册实例标识，用于拒绝失效或不匹配的句柄。一次
vendor read 所需参数由 `DirectReadOp` 传递：

```cpp
struct DirectReadOp {
    XdsFileRegistration file;
    XdsBufferRegistration buffer;
    uint64_t file_offset = 0;
    uint64_t device_offset = 0;
    uint64_t length = 0;
    int32_t device_id = -1;
};
```

必须满足：

```text
buffer.base == hipFileBufRegister 使用的原始 base
buffer.base + device_offset == 最终 dst
device_offset + length 不越过 buffer.length
file.id、buffer.id、device_id 与 adapter 当前注册表一致
```

buffer 的在途保护不嵌入 `DirectReadOp`，而是由 backend 的 `PlannedRequest::pins` 保存
`shared_ptr<XdsBufferEntry>` 并增加 `inflight`。file registration 由 `XdsFileEntry` 和
`shared_ptr<OpenFileHandle>` 保持有效；首版运行期间不淘汰已注册文件，backend 析构时统一
deregister 并释放 fd。

#### 3.2.4 `AcceleratorFileIo` 当前接口

```cpp
class AcceleratorFileIo {
  public:
    virtual ~AcceleratorFileIo() = default;

    virtual DirectIoCapabilities Capabilities() const = 0;
    virtual DeviceXdsState DeviceState(int32_t device_id) const = 0;

    virtual tl::expected<XdsFileRegistration, XdsError> RegisterFile(
        int fd, const XdsFileIdentity& identity) = 0;

    virtual tl::expected<XdsBufferRegistration, XdsError> RegisterBuffer(
        void* base, size_t length, int32_t device_id) = 0;

    virtual tl::expected<DirectReadCompletion, XdsError> Read(
        const DirectReadOp& op) = 0;

    virtual tl::expected<void, XdsError> DeregisterFile(
        XdsFileRegistration registration) = 0;

    virtual tl::expected<void, XdsError> DeregisterBuffer(
        XdsBufferRegistration registration) = 0;
};
```

`XdsFileRegistration` 包含进程内唯一 `id` 和 `fd`；`XdsBufferRegistration` 包含 `id`、
原始 base、length 和 device id。fd 所有权当前不转移给 adapter，而由
`XdsFileEntry::file` 的 `shared_ptr<OpenFileHandle>` 保持存活。adapter 只保存并校验 fd，
backend 析构时先 deregister vendor handle，再释放文件 keepalive。

### 3.3 Buffer registry

#### 3.3.1 entry 数据

当前 backend 中真实存在的 entry 为：

```cpp
struct XdsBufferEntry {
    uintptr_t base = 0;
    size_t length = 0;
    int32_t device_id = -1;
    pid_t pid = 0;
    uint64_t logical_refs = 1;
    uint64_t inflight = 0;
    bool accepting = true;
    bool quarantined = false;
    XdsBufferRegistration registration;
    std::condition_variable drained;
};
```

它存放在 `std::map<uintptr_t, shared_ptr<XdsBufferEntry>> xds_buffers_` 中，使用
`upper_bound` 查找覆盖目标地址的 allocation，不线性扫描所有 base。

当前 TE registration、`registered_buffer_sizes_` 和 xDS registry 分属现有层次。首版不把
它们合并为新的复合注册对象。

#### 3.3.2 注册规则

公开接口保持 `register_buffer(base, size)`。当前 `RealClient` 先执行
`Client::RegisterLocalMemory` 注册 Transfer Engine，再查询 pointer 的 memory kind 和
device id；device memory 才调用 `Client::RegisterXdsBuffer`。公开注册成功表示该区域至少
可继续服务现有 Transfer Engine/host fallback 路径，不表示它一定具备 xDS 能力。

规则：

- `base == nullptr` 或 `size == 0`：拒绝；
- `base + size` 溢出：拒绝；
- 部分重叠已有区域：拒绝；
- 相同 base、size、device、PID 的重复注册：只增加 logical refcount；
- 相同 base 但 size/device 不同：contract violation；
- fork 后 PID 改变：继承 entry 一律不可用；
- xDS 注册成功前不得写入 `xds_buffers_`，因此不会被 direct planner 查到。

通用事务顺序：

```text
Transfer Engine register
→ 查询 memory kind/device
→ device memory 且 mode != posix 时调用 RegisterXdsBuffer
→ backend 校验 interval/PID/device
→ hipSetDevice + hipFileBufRegister
→ 成功后写入 xds_buffers_
→ RealClient 记录 registered_buffer_sizes_
```

各模式行为固定为：

| 模式 | 单个 buffer 的 xDS 注册失败行为 |
|---|---|
| `posix` | 不调用 HyFile，保留现有 TE registration。 |
| `xds-auto` | xDS 注册失败时不创建 `XdsBufferEntry`，保留 TE registration，`register_buffer` 对调用方仍返回成功并记录日志。 |
| `xds-required` | xDS 注册失败时回滚 Transfer Engine registration，`register_buffer` 返回失败。 |

不支持的 memory kind、device 不可用以及确定失败的 `hipFileBufRegister`，在
`xds-auto` 下均属于 TE-only 降级。当前代码不会为 TE-only buffer 保存
`XdsIneligibleReason` entry，也不会在每次 range read 时临时重试；恢复 xDS eligibility
必须显式重新注册或重建 backend/device。adapter 按当前 vendor 契约把注册调用返回错误视为
`target_state=kUntouched`；这一契约必须由部署所用 `libhyfile.so` 的验收结果确认。

#### 3.3.3 pin

planner 查找 `[dst, dst + length)` 时：

1. entry 必须 `accepting == true` 且 `quarantined == false`；
2. range 必须完全位于一个 entry；
3. PID、device id 和可选 `registration_id` 必须匹配；
4. 增加 `inflight`；
5. 返回 `shared_ptr<XdsBufferEntry>` 作为当前 pin。

pin 生命周期覆盖：

- xDS read；
- read 后 device synchronize；
- 当前 backend 中的错误分类和 direct op 执行。

当前 pin 在 backend 返回给 `RealClient` 前释放，因此不覆盖 `RealClient` 的 session 发布
复查和之后的 fallback 决策。buffer 注销安全依赖 `Read()` 返回前已经完成
`hipDeviceSynchronize`，此时 direct DMA 已结束。

#### 3.3.4 解除注册

```text
logical_refs--
→ 非零：返回成功
→ 归零：accepting = false
→ 等待 inflight == 0
→ hipSetDevice + hipDeviceSynchronize
→ hipFileBufDeregister(base)
→ 成功后从 xds_buffers_ 删除 entry
```

随后 `RealClient::unregister_buffer_internal` 再注销 Transfer Engine registration，并从
`registered_buffer_sizes_` 删除记录。TE-only buffer 在 backend 中没有 entry，xDS 注销
直接成功。`hipFileBufDeregister` 或其前置同步失败时，entry 进入 `quarantined` 且不从
registry 删除；当前调用返回错误，entry 保持 quarantined，禁止后续 direct read。

#### 3.3.5 SGLang 生命周期

`MooncakeDirectLinker.register_buffers()` 遍历 storage allocation 并注册整块 GPU KV pool。
`close()` 先停止并 join load/offload/read-plan 线程，再调用 `storage.close()`；当前 Python
`MooncakeStore.close()` 没有逐项注销接口，最终由 C++ 对象和 backend 析构执行清理。首版
要求 KV pool 的实际显存生命周期长于 Direct Linker 和 Mooncake storage 对象，不能在这些
对象仍可能执行读取时提前释放或复用显存。

### 3.4 Session 和可见性

#### 3.4.1 当前 session state

当前实现为：

```cpp
std::unordered_map<std::string, QueryResult> get_sessions_;
```

`batch_get_session_start` 查询并选择一个完整 replica，把过滤后的 `QueryResult` 写入 map；
同 key 再次 start 会替换旧 entry。range read 在锁内复制 `QueryResult`、descriptor 和
lease deadline 到 `SessionRangeReadRequest`，I/O 期间不持有 session mutex。

完成后通过 `CompatibleSessionCacheRefresh(old, current)` 比较 replica 身份和可用
checksum，并重新检查 lease。

#### 3.4.2 提交前检查

创建 `SessionRangeReadRequest` 快照时检查：

- session 存在；
- replica 完整且类型受支持；
- lease 未过期；
- range 不越过对象边界。

#### 3.4.3 完成前发布检查

direct、session-cache H2D 和 host fallback 完成后，在填写成功返回码前再次检查：

```text
当前 key 仍存在且 replica 与发起 I/O 时兼容
lease 未过期
所有设备操作已完成
```

失败时不发布成功。只要 device synchronize/H2D synchronize 已成功，目标内存可由上层按失败路径安全回收或覆盖。

#### 3.4.4 session end

`batch_get_session_end` 在 `session_mutex_` 下删除 map entry 和 prefetch cache，不等待
in-flight get。首版调用契约要求同一个 key 遵守：

```text
batch_get_session_start
→ 一个或多个 range read 完成并返回
→ batch_get_session_end
```

同一 key 的 `start`、range read 和 `end` 不得并发，也不得在旧 range read 未返回时重新
start 同名 key。完成前的 replica compatibility 和 lease 复查用于防止已经失效的读取结果
被报告成功，但不替代该调用顺序契约。

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
复制 session/replica 快照并校验 ranges
  → get_session_prefetch_cache_ 命中？
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
range 完全位于 accepting 且未 quarantined 的 XdsBufferEntry
registration id、PID 和 device 有效
registration device 与 request device 一致
同一 key 的 ranges 属于同一 device（首版限制）
file_offset = value_offset + object_offset 无溢出
file_offset % 4096 == 0
uintptr(dst) % 4096 == 0
length % 4096 == 0
0 < length <= SSIZE_MAX
coalescing 后每个 DirectReadOp.length >= MIN_READ_SIZE
```

普通 eligibility 不满足属于 `kNotEligible`，直接加入 host fallback，不调用 HyFile，不计 backend 故障。

#### 4.1.4 range coalescing

首版只合并 request 中相邻、同时满足以下条件的 range：

```text
相同 file registration
相同 buffer registration
相同 device
前一 file range 末尾 == 后一 file range 起点
前一 device range 末尾 == 后一 device range 起点
合并后 length <= SSIZE_MAX
合并后仍位于同一对象和 registration 内
```

不为了合并而跨对象读取，不读取相邻 key 数据，不重排可能改变覆盖关系的 range。

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
- `resolved_value_offset = descriptor.offset`。

当前 BUCKET-first 首版不对 SHARD 执行 xDS：`GetXdsFile` 在非 BUCKET allocator 下返回
`kNotEligible/allocator-not-supported`，由 `RealClient` 按模式和安全策略走原 host 路径。

#### 4.2.3 BUCKET

- `descriptor.shard_idx` 被解释为 bucket id，必须非负并可安全格式化；
- backend 在 `canonical_root_dir` 下根据 bucket id 重建预期 data path；
- 不直接信任 descriptor 中的任意路径打开文件；
- 使用当前 `BucketEntryLayout` 验证 offset、object size、aligned size 和 bucket capacity；
- `resolved_value_offset = descriptor.offset`；
- BUCKET 文件按需注册并放入有上限的 `xds_files_` registration cache。
- 达到 `MOONCAKE_XDS_FILE_CACHE_SIZE` 后返回 `xds-file-cache-full`，由上层按模式和策略
  fallback 或报错；首版不在运行时淘汰已注册文件。

#### 4.2.4 File identity 和 cache

当前 `XdsFileIdentity` 包含：

```text
canonical reconstructed path
st_dev
st_ino
```

禁止仅以 POSIX fd 数字作为 identity，因为 fd 会被复用。
allocator type 和 bucket id 由 resolver 与 canonical path 间接约束。

xDS direct fd 与 fallback fd 分离：

```text
xDS:      O_RDONLY | O_DIRECT | O_CLOEXEC
fallback: 现有 buffered/direct adapter 自己管理的 fd
```

当前缓存项持有 `shared_ptr<OpenFileHandle>`，并使用 path + `st_dev/st_ino` 防止 fd/path
身份变化。缓存项在 backend 析构时先调用 `DeregisterFile`，随后释放文件 keepalive。

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

成功后 adapter 的 `files_` 以 registration id 保存 vendor handle/fd，backend 的
`XdsFileEntry` 通过 `shared_ptr<OpenFileHandle>` 保持 fd 存活。

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

auto file_handle = files_.at(op.file.id).handle;
void* registered_base = buffers_.at(op.buffer.id).base;
ssize_t ret = api.hipFileRead(
    file_handle,
    registered_base,
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

任何短读或 vendor error 都视为失败。保持当前 file handle 和 buffer `inflight` 保护，先执行
`hipDeviceSynchronize()`：

```text
synchronize success
    → 根据已确认的手工 ABI 映射错误类别
    → target_state = kStoppedSafeToOverwrite

synchronize failure
    → kBackendFatal
    → target_state = kUnknownMayStillBeWritten
    → backend sticky fatal，禁止 fallback 和后续 xDS read
```

首版不在 HyFile adapter 层自动重试 `hipFileRead`，避免目标状态不清时重复写入。

### 5.2 完整读取执行流程

#### 5.2.1 初始化

当前初始化流程为：

```text
解析配置
→ dlopen/dlsym libhyfile.so
→ 验证 7 个 required symbols
→ hipFileDriverOpen
→ hipGetDeviceCount
→ 将枚举到的 device 初始标记为 kAvailable
→ 创建 HyFileAcceleratorFileIo 并注入 DistributedStorageBackend
```

`xds-auto` 初始化失败时不注入 adapter并继续 POSIX；`xds-required` 初始化失败时 backend
创建失败。运行 xDS 业务前，部署侧必须先完成 7.1 的真实 mount、逐 device 独立验收；
Mooncake 初始化流程不重复执行该验收。

#### 5.2.2 一次 batch

首版为便于证明安全，执行顺序固定为：

1. 校验所有参数、session、对象边界和 batch 目标地址无重叠。
2. 查询 `get_session_prefetch_cache_`，为命中项构建 host-source H2D plan，但暂不提交。
3. 对 cache miss 做 direct eligibility 和 coalescing。
4. 对所有 direct key 增加 buffer entry 的 `inflight` 并保存 `shared_ptr`；文件由
   `XdsFileEntry` 和 `OpenFileHandle` keepalive 持有。
5. 在专用 worker 或当前同步上下文中逐个执行 direct ops；HyFile 内部仍全进程串行。
6. direct key 全部 op 成功后标记为 direct-complete 候选。
7. direct key 任一 op 失败：
   - `kUntouched` 或 `kStoppedSafeToOverwrite` 且策略允许：把整 key 加入 fallback；
   - `kUnknownMayStillBeWritten`：禁止 fallback，backend 进入 sticky fatal；
   - contract violation：请求失败，不掩盖为普通 fallback。
8. direct 阶段结束后，只为 initial fallback 和 safe failed-direct keys 分配 host arena。
9. 调用现有 `Client::BatchGet` 读取完整对象。
10. 为 session-cache 和 fallback key 构建并提交 `DfsAsyncScatterContext`。
11. 同步所有已使用 H2D streams。
12. 对所有成功候选执行 replica compatibility 和 lease 发布检查。
13. `RealClient` 通过每个计划项保存的 `original_idx` 回填原始结果数组；成功值为该原始 key 的 `sum(all_sizes[original_idx])`，失败值为稳定负错误码。
14. 释放 buffer `inflight`/`shared_ptr`、source handle 和 arena。

首版不并行执行 xDS 与 H2D，保持上述可证明的顺序和故障语义。

#### 5.2.3 session cache

- 有效 `get_session_prefetch_cache_` 完整对象命中优先于 xDS，继续走 H2D；
- cache source `shared_ptr<BufferHandle>` 持有到 H2D stream synchronize 完成；
- xDS range 成功不创建伪造 host handle，不填充完整对象 cache；
- fallback 完整对象成功后，可以按现有策略进入 session cache；
- 当前 session/replica 不再兼容时不得发布或插入 cache；
- `batch_get_session_end` 删除 map entry，但不能提前释放被在途 H2D 持有的 source。

#### 5.2.4 返回语义

一个 key 只有在以下条件全部满足时返回请求字节数：

```text
该 key 的全部 ranges 已完成
direct synchronize 或 H2D synchronize 成功
没有 range 失败
当前 replica 快照仍兼容
lease 未过期
目标 registration id 仍有效
```

混合 batch 中某个 key 的失败不自动覆盖其他 key 的结果。当前 backend fatal 会设置
sticky `xds_fatal_` 并拒绝后续 xDS。服务恢复依赖外部 supervisor 重启承载进程。

返回与索引映射必须满足：

- `AcceleratorFileIo::Read` 返回单个物理 op 的实际字节数，只用于 exact-length 校验；
- `DistributedStorageBackend::BatchReadRanges` 和 `Client::BatchReadDfsRanges` 返回与输入等长、同序的逐 key `expected<void, XdsError>`；
- backend 内部 coalescing 使用局部 `request_index` 把每个 op 的结果聚合回 key，不接触公开 `original_idx`；
- `RealClient` 的 direct、initial fallback、safe failed-direct fallback 子计划均保存 `request_index -> original_idx` 映射，最终只有 `RealClient` 计算业务成功字节数；
- 任一子层返回数量不匹配时，该子批次全部以 `INTERNAL_ERROR` 失败；
- 每 key 请求字节总和超过 `INT_MAX` 时在提交任何 I/O 前返回 `INVALID_PARAMS`。

### 5.3 对齐处理

首个版本只支持完全对齐 direct range：

```text
file_offset % 4096 == 0
uintptr(dst) % 4096 == 0
length % 4096 == 0
```

不对齐 key 走 host fallback。

## 6. 错误处理、回退与运行时恢复

### 6.1 错误和 fallback

#### 6.1.1 阶段化处理

| 失败阶段 | 目标状态 | 首版行为 |
|---|---|---|
| 参数、range、registration eligibility | untouched | host fallback 或参数错误 |
| `dlopen`/`dlsym` | untouched | auto 禁用；required 启动失败 |
| driver open | untouched | auto 禁用；required 启动失败 |
| direct fd open/file register | untouched | 策略允许时该 key fallback |
| buffer register | untouched | registration 不发布，启动/注册失败 |
| `hipSetDevice` before read | untouched | contract/fatal，禁止调用 read |
| `hipFileRead` 短读或错误，sync 成功 | stopped-safe | 整 key fallback |
| `hipFileRead` 后 sync 失败 | unknown | backend sticky fatal，禁止 fallback，当前请求失败 |
| 成功 read 后 sync 失败 | unknown | backend sticky fatal，禁止 fallback，当前请求失败 |
| H2D fallback 失败 | stopped | key 失败，不发布 |
| session/lease 发布检查失败 | stopped | 返回 session/lease 错误，不发布 |

#### 6.1.2 不允许 fallback 的错误

- descriptor/path 校验失败；
- object range 越界；
- batch 目标地址重叠；
- session 不存在或当前 replica 快照不兼容；
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

### 6.2 Hung I/O 和进程恢复

首版 vendor ABI 没有经过验证的 request timeout/cancel。进入 `hipFileRead` 后不能从另一个线程强制 deregister buffer 或 file。

当前 Mooncake 不提供 vendor request timeout/cancel，也不在进程内尝试恢复未知 DMA 状态。
生产部署应由现有外部 supervisor 监控承载进程；发现进程卡死或 backend fatal 后终止并
启动新进程，由新进程重新初始化 driver、file handle 和 buffer registration。本文不规定
额外的 Mooncake watchdog 或健康状态机实现。

deadline 根据真实 DFS 尾延迟确定，不能把普通慢 I/O 误判为 hung。该阈值属于部署配置，不由 `AcceleratorFileIo` 自行中断系统调用。

### 6.3 Fork 和多 device

- driver、file handle、buffer registration 都是进程资源；
- 当前进程级 runtime、buffer entry 和 adapter 操作保存/检查创建 PID；
- 检测到 PID 改变时，不调用继承资源的 vendor deregister；
- 子进程把继承对象标记为 abandoned，清空可查询 registry，并拒绝继续使用继承的 HyFile runtime；
- 由 fresh exec 启动的新 worker 从 `hipFileDriverOpen`、buffer registration 和 file registration 完整重建；
- 每个 device pool 分别注册并保存 device id；
- 每次 register/read/synchronize/deregister 前设置正确 device context；
- 首版同一 key 的 ranges 必须位于同一 device；一个 batch 可以包含不同 device 的不同 key，但 HyFile API 仍全进程串行。

## 7. 初始化、配置与写路径边界

### 7.1 独立部署验收

真实 DFS mount 与 `libhyfile.so`/驱动/device 的组合必须在启用业务 xDS 前通过独立验收。
该验收不是 Mooncake 初始化流程的一部分，不增加环境变量、运行时状态机或业务请求内 probe。

对每张业务会使用的目标 DCU、每个目标 Parastor/HYFS mount，独立验收程序执行：

1. 分配并注册 4096 对齐、长度至少 4096 的 device buffer；
2. 在真实 mount 上以唯一临时文件名创建文件，用 POSIX exact-length write 写入已知数据，
   完成必要同步后关闭写 fd；
3. 以 `O_RDONLY | O_DIRECT | O_CLOEXEC` 重新打开 read fd；
4. 把 read fd 注册为 opaque fd file handle；
5. 使用 `hipFileRead` 把数据读入目标 device buffer；
6. `hipDeviceSynchronize`，再 D2H 并逐字节比较；
7. 逆序注销 buffer/file、关闭 read fd 并删除临时文件。

验收报告至少记录 DCU、DTK、`libhyfile.so` build-id、驱动、DFS client、mount 参数、device
id、对齐值和结果。任一目标组合失败时不得在该组合上启用 `xds-auto` 或
`xds-required`。独立程序可复用 `hyfile_abi.h` 和 adapter 代码，但不形成第二套业务实现。

### 7.2 业务写路径

#### 7.2.1 首版

| 操作 | 行为 |
|---|---|
| DFS → DCU KV | 实现 xDS direct read |
| DCU KV → DFS | 保持现有 D2H staging + `BatchWrite` |
| `hipFileWrite` | 首版不加载、不调用 |
| 生产 direct write | 不支持 |

业务 `DistributedStorageBackend::BatchWrite` 不得调用 `AcceleratorFileIo::Write`；首版接口甚至不暴露业务 `Write` 方法。

业务 xDS write 不属于本文范围；如果后续需要，必须单独设计和评审，不能直接把
`hipFileRead` 替换为 `hipFileWrite`。

### 7.3 配置

配置优先级：显式程序配置 > 环境变量 > 默认值。

当前代码实际解析的配置：

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `MOONCAKE_DFS_DIRECT_IO_BACKEND` | `posix` | `posix`、`xds-auto`、`xds-required` |
| `MOONCAKE_XDS_LIBRARY_PATH` | `/opt/hyhal/lib/libhyfile.so` | 显式动态库路径 |
| `MOONCAKE_XDS_ALLOW_REQUEST_FALLBACK` | `true` | safe request failure 是否允许整 key fallback |
| `MOONCAKE_XDS_MIN_READ_SIZE` | `16384` | 每个 coalesced DirectReadOp 的最小长度 |
| `MOONCAKE_XDS_STAGING_BUFFER_BYTES` | `4194304` | 每个实际使用 device 懒分配并复用的 GPU staging 大小；必须为正的 4096 字节整数倍 |
| `MOONCAKE_XDS_FILE_CACHE_SIZE` | `1024` | BUCKET xDS file registration cache 上限；达到上限后不再注册新文件 |

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
    初始化失败 → 使用 POSIX
    普通不适用请求 → POSIX fallback
    safe transient request failure → 策略允许时 fallback
    backend fatal → sticky 拒绝后续 xDS，当前请求失败，不做 host fallback

xds-required:
    初始化失败 → 启动失败
    不适用请求 → 只有 ALLOW_REQUEST_FALLBACK=true 才允许 fallback
    backend fatal → sticky 拒绝后续 xDS，当前请求失败，不做 host fallback
```

当前启动日志输出最终 `DistributedStorageConfig` 和 adapter 初始化错误，不输出业务数据。

## 8. 可观测性与测试

### 8.1 可观测性

首版复用现有日志和请求返回码，至少能够区分 adapter 初始化失败、request ineligible、safe
fallback、contract violation、short read、device synchronize 失败和 backend fatal。日志不得记录
完整 KV key、payload 或其他敏感业务数据。部署验收报告与 Mooncake 运行日志分开保存。

端到端评估继续使用现有 SGLang prefetch latency、TTFT、ITL、CPU 使用率、host memory
bandwidth 和 DCU copy-engine 利用率。新增专用 metrics 不作为首版路径正确性的前置条件。

当前批次日志分别汇总 direct 与 GPU staging 的逻辑字节数、物理读取字节数、
`hipFileRead` 调用数和完成 op 数；staging 路径还记录 chunk 数与最大逻辑 op 大小。
这些字段用于评估对齐读取放大，并判断 `MOONCAKE_XDS_STAGING_BUFFER_BYTES` 是否需要调优。

#### Staged 聚合候选只读统计

下一轮观测可在承载 Mooncake backend 的进程启动前设置
`MOONCAKE_XDS_STAGED_AGGREGATE_DRY_RUN=1`（默认 `0`）。这只是统计开关，
**自身不跳过真实 I/O、不重排或合并真实 operations**，也不改变目标状态、pin 生命周期或
fallback。关闭此开关或调用方不请求 stats 时，不计算额外候选统计；独立真实聚合开关
开启时仍会构造执行窗口，不依赖 stats 指针。

每次 `Client::BatchReadDfsRanges` backend 调用最多输出一条
`[XDS_STAGED_AGG_DRY_RUN] scope=planned`，不采样、不记录逐 range 或原始 key。
它与 `[XDS_READ_SUCCESS]` 独立：计划可能随后失败或因 fatal 未执行
（`includes_failed_or_unexecuted_plans=1`），不可当作成功 I/O 或已实现的性能收益。

- `evaluated_objects` / `skipped_objects`：已评估 / 跳过的请求对象数；无效整批不评估，
  单对象规划失败则跳过该对象。`staged_ops` 是现有 planner 合并后的 staged op 数，
  不含 direct ops；`candidate_windows` 是被接受的多目标窗口数。
- `baseline_calls` / `candidate_calls` 及 `baseline_physical_bytes` /
  `candidate_physical_bytes`：原计划 / 模拟计划的 vendor calls 与物理字节；
  `saved_calls` / `saved_physical_bytes` 是二者差值，仅表示已评估子集。
- `scratch_bytes` 来自 adapter capabilities 的实际容量，而不是配置猜测；容量未知
  （`0`）或不是 4K 整数倍时，`scratch_unavailable=1`，跳过估计。
- `gap_rejections` / `capacity_rejections` 计数贪心扫描中因正 gap / 联合覆盖超容量
  而结束当前窗口的次数，不是所有两两组合数；`oversized_ops` 是单 op 覆盖超容量数。
- `estimate_valid=0` 表示未能完整估计可评估计划；`arithmetic_overflow` /
  `observation_failed` 标识统计算术 / 资源异常。失败对象不提交部分计数，也不改变 I/O。
  汇总候选收益时必须筛选 `estimate_valid=1`；无效行可能保留此前对象的部分统计，
  不能当作完整 batch 的估计，也不能将缺失估计当作零收益。

基线使用 4K 覆盖 `P = align_up(offset + length, 4096) - align_down(offset, 4096)`，
以实际 scratch 容量 `S` 计算 `calls = P / S + (P % S != 0)`，不是把 op 数当 calls。
仅在同一请求/object、文件、device、extent 内对源覆盖窗口副本排序，贪心合并重叠或
相接且联合覆盖不超过 `S` 的窗口，不跨正 gap、对象 extent 或已验证 EOF。只有 calls
下降且 bytes 不增加才接受候选；各最终窗口的物理字节分别求和，不能用全局 union
替代。超容量 op 保留原有分块 calls/bytes，不参与聚合且不枚举 chunks，因此这是保守
候选估计而非全局最优。不额外打开文件或执行 `fstat`。关闭开关即可停止统计；本变更
不会自动启动负载或部署远端。

#### 真实 staged 源窗口聚合与 scatter（路线 A）

`MOONCAKE_XDS_STAGED_AGGREGATE_READ=1` 独立开启真实聚合，默认 `0`；DRY_RUN
只控制观测，两开关可任意组合。均关闭保持原有执行行为。真实聚合与 dry-run 使用同一
源排序贪心规划函数，dry-run 始终以聚合前的原 operations 为基线。

仅同 request/object、file、device、extent 的 staged 成员可组成窗口，4K 覆盖必须
重叠或相接、联合长度不超过 adapter 实际 scratch。独立 `supports_staged_window`
capability 才授权调用 `ReadStagedWindow(vector<DirectReadOp>)`，不能用已有 vendor
scatter/gather 字段替代。未支持、容量未知、单目标或超容量 op 在调用前保留旧路径
（超容量仍按旧分块执行）；一旦调用窗口接口，失败不能重试单目标 staged 读取。

所有 batch 参数校验及目标 pin、所有窗口构造均在首次 I/O 前完成。窗口放在其成员
最小原 op index 处执行，成员按原顺序 scatter，其他 operations 保持彼此相对顺序。
成员无需连续：这可能提前执行后续成员，安全性依赖既有全 batch 目标内存不重叠验证
和请求完成后发布契约，并不保证开启后完全维持逐 op 时序。

HyFile 单/多目标共用内部 pointer/count 执行器与原 scratch、锁、quarantine。
全部成员的注册、bounds、file/device/extent、一致性、覆盖连通、目标不重叠和 logical
sum 溢出检查先于 I/O，并再次检查实际 EOF。一次窗口 read/sync 后逐目标 D2D，每次
copy 后同步；不增加 stream、scratch 或并发。失败不计该窗口部分成功；未知目标状态
保留相关请求全部 pins，fatal 阻止后续执行，不提前发布请求成功。

`[XDS_READ_SUCCESS] path=gpu_staging` 中：

- `completed_ops` / `staging_completed_ops` 计完成的 backend **执行单元**：窗口计 1，
  不是目标成员数，也不是 vendor call 数。
- `aggregate_windows` 计完整成功的聚合窗口，`aggregate_targets` 计这些窗口的原 op
  成员数；单目标旧路径不计入两者。
- `logical_bytes` 是所有目标有效长度之和，源重叠也分别计；`physical_bytes` 对每个
  窗口只计一次物理读取。`max_op_bytes` / `max_staged_op_bytes` 为执行单元的 logical
  sum，源重叠时可以大于 scratch 容量，不能再据此推断物理窗口大小。
- 失败窗口不增加成功 counters，即便较早目标已经写入；错误目标状态独立传播。

验证应先在 Linux/HIP 环境编译并运行 `hyfile_backend_test`、
`xds_buffer_registry_test`、`dfs_posix_test`，再由用户控制同配置、同缓存条件的 48 并发
A/B；比较 vendor calls/physical bytes、D2D/同步开销、恢复耗时、端到端延迟及失败率。
候选统计不是实测加速。本变更不自动部署、重启或运行负载。

### 8.2 测试方案

当前已经存在：

- `hyfile_backend_test.cpp`：缺库/缺符号、driver 单次开关、buffer 注销同步、quarantine、
  exact read、短读、同步失败和 fork 防护；
- `xds_buffer_registry_test.cpp`：重复逻辑注册、部分重叠拒绝和 batch 目标重叠拒绝。

下列各小节是完整验收测试矩阵；未在上述文件或其他现有测试中覆盖的条目仍需实现。

#### 8.2.1 Planner 单元测试

- SHARD 和当前 BUCKET `file_offset` 计算；
- 空 ranges、零长度、null dst；
- object offset/length/file offset/address 的整数溢出；
- object range 越界；
- batch 重复 key；
- 同 key和跨 key目标地址重叠；
- 未注册、跨 registration、`accepting=false` 的 registration；
- PID 或 registration id mismatch；
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
- `posix` 不创建 xDS entry，现有 TE registration 正常工作；
- `xds-auto` 中安全的 xDS 注册失败不创建 xDS entry，TE registration 保留且公开注册成功；
- `xds-required` 中 xDS 注册失败回滚 TE registration 且公开注册失败；
- unsupported memory kind 在 auto 模式下只保留 TE registration；
- pin 与 unregister 并发；
- `accepting=false` 后拒绝新 pin；
- inflight 归零后才 deregister；
- deregister 失败进入 quarantined 并返回错误；
- fork/PID 改变后不调用继承资源的 vendor cleanup；
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
- buffer `inflight` 生命周期、file handle keepalive 与 fd identity；
- file cache 达到上限后返回 `xds-file-cache-full`；
- backend 返回结果与输入等长同序，内部 coalescing 后仍能按局部 request index 聚合；
- 返回数量不匹配时子批次全部得到 `INTERNAL_ERROR`；
- `RealClient` 使用 `original_idx` 回填混合 direct/fallback batch，并且只有它计算公开成功字节数；

#### 8.2.4 Session 和路由测试

- session cache hit 优先；
- cache miss direct；
- initial ineligible fallback；
- safe failed-direct 整 key fallback；
- direct 不分配 host arena、不提交 H2D、不写完整对象 cache；
- fallback 成功后按现有配置写 session cache；
- start → range read → end 的正常顺序；
- session 或 replica 在完成检查时已变化则不发布成功；
- lease 在 I/O 中过期时不发布成功；
- mixed batch 等待全部 direct/H2D completion 后返回；
- backend fatal 后 sticky 拒绝后续 xDS 请求。

#### 8.2.5 真实 DTK/xDS 集成测试

在真实海光节点和 Parastor/HYFS mount 上覆盖：

1. 按 7.1 对每个目标 device/mount 完成独立 round-trip 验收；
2. 单文件、单 device、单 aligned range；
3. 单文件、多 range；
4. 多文件和 file cache 达到上限；
5. 多 device batch；
6. BUCKET fully-aligned range；
7. SHARD 请求确认为 ineligible 并按策略 fallback；
8. 不对齐 file/device/length 的 fallback；
9. `libhyfile.so` 缺失和 required symbol 缺失；
10. runtime/driver 版本不匹配；
11. direct fd 必须包含 `O_DIRECT`，fallback 使用独立 fd；
12. fork 后由 fresh exec 完整重建；
13. handle/buffer 重复注册和解除注册；
14. read 短返回和错误注入；
15. synchronize 失败使 backend sticky fatal，目标状态 unknown 时禁止 fallback；
16. 多线程压力下仍只有一个 HyFile 临界区；
17. 24 小时 soak 检查 fd、registration、显存和 driver 资源。

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
- 多 BUCKET 文件和 file cache 达到上限后的 fallback/错误；
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
| allocator | BUCKET |
| path | POSIX buffered、O_DIRECT host、xDS direct |
| device count | 1、2、4、8 |
| SGLang mode | layer-wise、page-wise、ReadPlan |

特别记录每个 `hipDeviceSynchronize` 对同设备无关 compute 的影响。

## 9. 实施计划、代码改动与构建

### 9.1 分阶段实施计划

当前代码已经一次性接入 adapter、registry、BUCKET planner 和 `RealClient` 热路径。阶段
只用于表达验证顺序，不再为未确定的优化预设额外实现。

| 阶段 | 当前状态 | 仍需完成 |
|---|---|---|
| Phase 0 | 最小手工 ABI、动态加载、真实 HyFile adapter、fake 函数表测试已完成 | 目标硬件/真实 mount 独立 round-trip、版本矩阵和错误语义确认 |
| Phase 1 | 接口、buffer interval registry、有界文件缓存、planner、overlap/fork/错误处理已接入 | 补齐 8.2 中与当前实现直接对应的单元测试 |
| Phase 2 | BUCKET fully-aligned 软件调用链、同步 read、safe fallback、sticky fatal 已完成 | 真实硬件 E2E、零 host payload staging 验证、soak 和性能验收 |

一次性交付不表示可以跳过中间验证。验收顺序固定为：

```text
Phase 0 → Phase 1 → Phase 2
```

前一阶段退出条件未满足时不能启用后一阶段的数据面；默认 backend 始终为 `posix`。
Phase 2 是 BUCKET 场景首个真正打通 xDS read 的阶段。只有 Phase 0 的独立验收和 Phase 2
的真实硬件 E2E 均通过，才能称为可用并开始受控灰度。

#### Phase 0：ABI 和独立 PoC

- 固化并评审 `hyfile_abi.h` 的最小手工 ABI 声明和版本矩阵；
- 确认 `libhyfile.so` 路径和 required symbols，记录版本/build-id 但不做硬编码拒绝；
- 确认 `hipFileRead` 精确原型和错误语义；
- 在生产目录中实现可复用的 `hyfile_abi.h` 和 `HyFileAcceleratorFileIo`；
- 独立验收程序复用上述生产模块，不维护第二套 vendor 调用实现；
- 独立程序验证 O_DIRECT opaque fd；
- 每 device 整 pool/subrange 注册语义；
- POSIX 写入已知数据，xDS exact-length read，并逐字节校验；
- 4096 对齐；
- 进程级串行压力；
- fork 子进程拒绝继承的 vendor 资源，由 fresh exec 的新 worker 重建；
- hung-I/O 和 synchronize 语义。

退出条件：在目标版本矩阵上稳定完成逐字节正确的已知数据读回，错误语义明确，且验收
程序确认 payload 未经过 Mooncake host staging。

#### Phase 1：内部抽象和生命周期

- `AcceleratorFileIo`；
- fake backend；
- buffer interval registry 的 `shared_ptr + inflight` 生命周期；
- BUCKET canonical path、file identity 和有界 xDS registration cache；
- batch target-overlap validation；
- planner；
- session snapshot/replica/lease 发布检查；
- safe fallback 和 sticky fatal 错误模型。

退出条件：接口、规划、生命周期和故障路径的单元测试通过。

#### Phase 2：BUCKET fully-aligned read-only MVP

- 固化并启用 Phase 0/1 的手工 ABI 声明和 `HyFileAcceleratorFileIo`；
- 新增 `BatchReadDfsRanges`；
- 接入 `batch_get_into_multi_buffer_ranges`；
- 只支持 BUCKET、完全对齐、达到阈值的 key；
- 复用当前对齐的 `BucketEntryLayout`，不改变磁盘格式；
- bucket id → canonical path 重建和 descriptor/range 校验；
- BUCKET xDS file registration cache、fd identity 和 buffer inflight；
- cache 达到上限时按模式和策略 fallback 或报错；
- 同步 read、global mutex、exact-length、per-op device synchronize；
- safe failure 整 key fallback；
- unknown target state 禁止 fallback，并将 backend 置为 sticky fatal。

退出条件：真实 BUCKET 硬件 E2E correctness、fallback、sticky fatal、零 host payload staging、
file cache 上限、fork/fresh-exec、灰度开关和 soak 全部通过。达到此条件即视为 BUCKET xDS
read 路径真正可用，可开始受控灰度。

SHARD direct read、不对齐 bounce、异步/batch 和业务 xDS write 均不属于本实施计划；若有
明确需求，另行建立设计文档和验收标准。

### 9.2 代码改动清单

当前实现的主要代码落点：

| 当前文件 | 已实现入口 |
|---|---|
| `real_client.cpp/.h` | `register_buffer_internal`、`batch_get_session_start_with_sources`、`batch_get_into_multi_buffer_ranges`、`execute_session_dfs_range_reads`、`batch_get_session_end` |
| `client_service.cpp/.h` | `RegisterXdsBuffer`、`UnregisterXdsBuffer`、`BatchReadDfsRanges` |
| `distributed_storage_backend.cpp/.h` | `RegisterXdsBuffer`、`UnregisterXdsBuffer`、`PinXdsBuffer`、`GetXdsFile`、`BatchReadRanges` |
| `accelerator_file_io.h` | vendor-neutral registration、capability、`DirectReadOp`、`XdsError` 接口 |
| `xds/hyfile_abi.h` | 唯一的最小手工 ABI 声明 |
| `xds/hyfile_backend.cpp` | runtime 获取、`dlopen`/`dlsym`、文件/显存注册、真实 read/sync、fork 防护 |
| `storage_backend.cpp` | 按 mode 创建并注入 HyFile adapter |
| `hyfile_backend_test.cpp` | adapter/runtime/fork/错误状态测试 |
| `xds_buffer_registry_test.cpp` | buffer registry 和 batch overlap 测试 |

`hyfile_abi.h` 是唯一的 vendor ABI 声明源，`hyfile_backend.cpp` 是唯一执行
`dlopen`/`dlsym` 和调用 vendor 函数的编译单元。

### 9.3 构建

建议构建参数：

```text
-DUSE_HIP=ON -DUSE_XDS=ON
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
- unknown target state 必定使 backend sticky fatal，且不执行 fallback；
- session replica 不再兼容或 lease 失效时不发布成功；
- target overlap 在任何 I/O 前被拒绝；
- fork 后不访问继承的 vendor 资源；
- 24 小时 soak 无 fd、registration、显存或 driver 资源增长。

#### 10.1.2 兼容性

- `MOONCAKE_DFS_DIRECT_IO_BACKEND=posix` 时行为与当前版本一致；
- xDS 构建关闭时不引入 vendor runtime 依赖；
- SGLang 公开 API不变；
- Master 元数据协议不变；
- 当前业务 DFS 写路径不变；
- auto 模式缺少库或符号时使用 POSIX；运行期 safe request failure 可整 key fallback。

#### 10.1.3 性能

平台和模型的数值门槛由真实硬件 E2E 和性能验收确定，但至少必须满足：

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
| `libhyfile.so` ABI 随 DTK 变化 | 单一手工 ABI 声明、最小 required symbols、部署版本清单和独立真实 mount 验收；不兼容时 fail closed |
| HyFile/driver 对目标 DFS 不支持或在内部静默 bounce | 逐 device/mount 独立验收；没有厂商 counter 时不宣称驱动内部零 bounce |
| DCU buffer 生命周期错误 | 整 pool eager registration、`shared_ptr + inflight`、unregister drain 和 quarantined 状态 |
| BUCKET 文件被替换或 FD 被复用 | canonical path + `st_dev/st_ino` identity、`OpenFileHandle` keepalive；缓存满时不淘汰已有项 |
| 大量小 range 的调用与同步开销超过收益 | `MIN_READ_SIZE`、双向连续 coalescing 和真实 workload 性能验收 |
| HyFile 非线程安全 | 首版使用进程级 mutex 串行全部 vendor API，实机并发压力测试 |
| 部分完成或未知 DMA 状态使错误 KV 可见 | exact-length、device synchronize、每 key 聚合结果；未知 target state 禁止 fallback 并使 backend sticky fatal |
| 同步 I/O 永久阻塞 worker | vendor 无 timeout/cancel；由部署已有 supervisor 检测并重启整个承载进程 |
| 注册显存或文件句柄资源过高 | 整 pool 注册、注册字节数指标、有界 file cache、启动 fail-fast 和 soak |
| direct path 绕过 descriptor、路径或 lease 校验 | 强制复用 canonical resolver，提交前和发布前分别校验 replica compatibility/lease |

### 10.3 部署前仍需厂商确认

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

未绑定实际函数的 IO params、event、poll 或 batch 结构一律视为影子 ABI，首版不得使用。
上述待确认项在厂商确认和实机验证前一律按保守策略处理。

### 10.4 首版最终契约

当前代码和后续验收均以以下范围为准：

- 业务 read-only xDS；
- BUCKET allocator；
- 仅通过 7.1 独立验收的 Parastor/HYFS mount/device 组合；
- `libhyfile.so` 显式 `dlopen`；
- `O_DIRECT` opaque fd；
- SGLang KV pool eager registration；
- 注册阶段解析 device id；
- 完全对齐且达到阈值的 ranges；
- bucket id → canonical path 重建和 descriptor/range 校验；
- BUCKET 有界 xDS file registration cache、canonical identity 和 fd keepalive；
- cache 满时不淘汰已有项，按模式和策略 fallback 或报错；
- buffer `shared_ptr + inflight` 和 unregister drain；
- batch key 唯一且目标地址不重叠；
- 每 key 全 direct 或全 fallback；
- 同步 `hipFileRead`；
- exact-length；
- 全进程 HyFile API mutex；
- 每 op 返回前 device synchronize；
- safe failure 整 key fallback；
- unknown target state 禁止 fallback并使 backend sticky fatal；
- 完成前 replica compatibility 和 lease 校验；
- 同一 key 的 session start/range read/end 不并发；
- 不实现非对齐 bounce；
- 不实现业务 xDS write；
- 不实现 async/batch/event/poll；
- 不实现进程内 timeout/cancel；
- `posix` 默认，灰度时显式选择 `xds-auto`；
- `xds-required` 用于验收和强制部署；
- 一键切回当前 POSIX/O_DIRECT + host staging 路径。

该范围可以验证核心价值：在不改变 SGLang 对外接口和 Master 协议的前提下，消除 fully-aligned DFS KV range 的完整对象 host staging 和 H2D scatter，同时对不适用和可恢复错误保留现有稳定路径。

### 10.5 参考资料

- 当前代码基线：`mooncake-das` commit `dded8f87051d537ba677193cb22941f564f1a330`，以及 `sglang-das` 的 `feature/optimize_the_direct_linker` commit `c4257ce7b190710b953504645aea1cdd7a53035e`。
- LMCache xDS/HyFile 适配代码审阅记录（本设计输入，2026-09）。
- [LMCache GDS backend documentation](https://github.com/LMCache/LMCache/blob/dev/docs/source/kv_cache/storage_backends/gds.rst)：用于对照 direct-storage backend 的配置和测试思路；HyFile ABI 由本项目的最小手工声明、部署版本清单和独立真实 mount 验收共同约束。
- [LMCache GDS context tests](https://github.com/LMCache/LMCache/blob/dev/tests/v1/gpu_connector/test_gds_context.py)：用于对照 GPU buffer/context 生命周期测试。
