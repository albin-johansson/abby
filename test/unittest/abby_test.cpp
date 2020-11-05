#include "abby.hpp"

#include <doctest.h>

TEST_SUITE("aabb_tree")
{
  TEST_CASE("aabb_tree::insert")
  {
    abby::aabb_tree<int> tree;
    REQUIRE(tree.is_empty());

    tree.insert(1, {{0, 0}, {100, 100}});
    CHECK(!tree.is_empty());
    CHECK(tree.size() == 1);

    tree.insert(2, {{40, 40}, {100, 100}});
    CHECK(tree.size() == 2);

    tree.insert(3, {{75, 75}, {100, 100}});
    CHECK(tree.size() == 3);
  }

  TEST_CASE("")
  {}
}
