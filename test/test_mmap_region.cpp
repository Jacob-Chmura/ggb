#include <sys/mman.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/mmap_region.h"

// Third-party
#include <gtest/gtest.h>

namespace fs = std::filesystem;
using ggb::detail::MmapRegion;

class MmapRegionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::ofstream ofs(test_file_, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(data_.data()),
              data_.size() * sizeof(float));
    ofs.close();
  }

  void TearDown() override {
    if (fs::exists(test_file_)) {
      fs::remove(test_file_);
    }
  }

  const std::string test_file_{"test_mmap_data.ggb"};
  const std::vector<float> data_{1.0, 2.0, 3.0, 4.0};
};

TEST_F(MmapRegionTest, MmapValidFile) {
  const MmapRegion mmap(test_file_);

  EXPECT_EQ(mmap.size(), data_.size() * sizeof(float));
  ASSERT_NE(mmap.data(), nullptr);

  const auto* const mapped_data = static_cast<const float*>(mmap.data());
  EXPECT_FLOAT_EQ(mapped_data[0], data_[0]);
  EXPECT_FLOAT_EQ(mapped_data[1], data_[1]);
  EXPECT_FLOAT_EQ(mapped_data[2], data_[2]);
  EXPECT_FLOAT_EQ(mapped_data[3], data_[3]);
}

TEST_F(MmapRegionTest, MadviseValidFile) {
  const MmapRegion mmap(test_file_);
  mmap.advise(MADV_SEQUENTIAL);
}

TEST_F(MmapRegionTest, MmapEmptyFile) {
  const std::string empty_file = "empty.ggb";
  std::ofstream ofs(empty_file);
  ofs.close();

  const MmapRegion mmap(empty_file);
  EXPECT_EQ(mmap.size(), 0);
  EXPECT_EQ(mmap.data(), nullptr);
}

TEST_F(MmapRegionTest, ThrowsOnMissingFile) {
  EXPECT_THROW({ MmapRegion mmap("non_existant.ggb"); }, std::runtime_error);
}

TEST_F(MmapRegionTest, MoveSemantics) {
  {
    MmapRegion mmap1(test_file_);
    const auto* const original_data = mmap1.data();
    const auto original_size = mmap1.size();

    const MmapRegion mmap2(std::move(mmap1));

    EXPECT_EQ(mmap2.data(), original_data);
    EXPECT_EQ(mmap2.size(), original_size);
  }
}
