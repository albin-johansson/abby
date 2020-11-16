#include <doctest.h>

#include "abby.hpp"

using node_t = abby::node<int, abby::vector2<double>>;

static_assert(std::is_same_v<node_t::key_type, int>);
static_assert(std::is_same_v<node_t::aabb_type , abby::aabb<abby::vector2<double>>>);

TEST_SUITE("node")
{
  TEST_CASE("node default values")
  {
    const node_t node;
    CHECK(!node.id);
    CHECK(node.aabb.min().x == 0);
    CHECK(node.aabb.min().y == 0);
    CHECK(node.aabb.max().x == 0);
    CHECK(node.aabb.max().y == 0);
    CHECK(node.height == -1);
    CHECK(!node.parent);
    CHECK(!node.left);
    CHECK(!node.right);
    CHECK(!node.next);
  }

  TEST_CASE("node::is_leaf")
  {
    SUBCASE("Leaf node")
    {
      const node_t node;
      CHECK(node.is_leaf());  // leaf node if left child is null
    }

    SUBCASE("Not leaf node")
    {
      node_t node;
      node.left = 123;
      CHECK(!node.is_leaf());
    }
  }
}
