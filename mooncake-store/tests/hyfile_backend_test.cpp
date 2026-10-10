#include <gtest/gtest.h>

#include "storage/distributed/xds/hyfile_backend.h"

namespace mooncake {
namespace {

TEST(HyFileBackendTest, MissingLibraryFailsClosed) {
    auto backend = CreateHyFileAcceleratorFileIo(
        "/path/that/does/not/exist/libhyfile.so");
    ASSERT_FALSE(backend);
    EXPECT_EQ(backend.error().error_class, XdsErrorClass::kNotEligible);
    EXPECT_EQ(backend.error().target_state, XdsTargetState::kUntouched);
    EXPECT_EQ(backend.error().operation, "dlopen(libhyfile)");
}

}  // namespace
}  // namespace mooncake
