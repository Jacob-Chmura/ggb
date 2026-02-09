#include <common/io.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Third-party
#include <gtest/gtest.h>

#include "ggb/core.h"

namespace fs = std::filesystem;

class MockBuilder : public ggb::FeatureStoreBuilder {
 public:
  struct Entry {
    ggb::Key key;
    ggb::Value values;
  };
  std::vector<Entry> received_;

  auto put_tensor_impl(const ggb::Key& key, const ggb::Value& tensor)
      -> bool override {
    received_.push_back({key, tensor});
    return true;
  }

  auto put_tensor_impl(const ggb::Key& key, ggb::Value&& tensor)
      -> bool override {
    received_.push_back({key, std::move(tensor)});
    return true;
  }

  auto build_impl([[maybe_unused]] std::optional<ggb::GraphTopology> graph)
      -> std::unique_ptr<ggb::FeatureStore> override {
    return nullptr;
  }
};

class IOTest : public ::testing::Test {
 protected:
  void create_csv(const std::string& content) {
    std::ofstream ofs(test_file_);
    ofs << content;
    ofs.close();
  }

  void TearDown() override {
    if (fs::exists(test_file_)) {
      fs::remove(test_file_);
    }
  }

  const std::string test_file_{"test_io_data.csv"};
};

TEST_F(IOTest, IngestFeatureCSV) {
  create_csv("1.0,2.0,3.0\n4.0,5.0,6.0\n");
  MockBuilder builder;

  ggb::io::ingest_features_from_csv(test_file_, builder);

  ASSERT_EQ(builder.received_.size(), 2);

  EXPECT_EQ(builder.received_[0].key, ggb::Key{.NodeID = 0});
  EXPECT_EQ(builder.received_[0].values.size(), 3);
  EXPECT_FLOAT_EQ(builder.received_[0].values[0], 1.0);
  EXPECT_FLOAT_EQ(builder.received_[0].values[1], 2.0);
  EXPECT_FLOAT_EQ(builder.received_[0].values[2], 3.0);

  EXPECT_EQ(builder.received_[1].key, ggb::Key{.NodeID = 1});
  EXPECT_EQ(builder.received_[0].values.size(), 3);
  EXPECT_FLOAT_EQ(builder.received_[1].values[0], 4.0);
  EXPECT_FLOAT_EQ(builder.received_[1].values[1], 5.0);
  EXPECT_FLOAT_EQ(builder.received_[1].values[2], 6.0);
}

TEST_F(IOTest, IngestsEdgeList) {
  create_csv("0,1\n1,2\n2,0");
  std::vector<std::pair<ggb::NodeID, ggb::NodeID>> edges;

  ggb::io::ingest_edgelist_from_csv(test_file_, edges);

  ASSERT_EQ(edges.size(), 3);
  EXPECT_EQ(edges[0].first, 0);
  EXPECT_EQ(edges[0].second, 1);
  EXPECT_EQ(edges[1].first, 1);
  EXPECT_EQ(edges[1].second, 2);
  EXPECT_EQ(edges[2].first, 2);
  EXPECT_EQ(edges[2].second, 0);
}
