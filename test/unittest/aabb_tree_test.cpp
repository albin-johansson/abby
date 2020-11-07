#include <doctest.h>

#include <iterator>

#include "abby.hpp"

TEST_SUITE("aabb_tree")
{
  TEST_CASE("aabb_tree::insert")
  {
    abby::aabb_tree<int> tree;
    REQUIRE(tree.is_empty());

    tree.insert(1, abby::make_aabb<float>({0, 0}, {100, 100}));
    CHECK(!tree.is_empty());
    CHECK(tree.size() == 1);

    tree.insert(2, abby::make_aabb<float>({40, 40}, {100, 100}));
    CHECK(tree.size() == 2);

    tree.insert(3, abby::make_aabb<float>({75, 75}, {100, 100}));
    CHECK(tree.size() == 3);
  }

  TEST_CASE("aabb_tree::emplace")
  {
    abby::aabb_tree<int> tree;

    tree.emplace(1, abby::vec2{1.0f, 1.0f}, abby::vec2{10.0f, 12.0f});
    CHECK(!tree.is_empty());
    CHECK(tree.size() == 1);

    const abby::vec2 position{89.3f, 123.4f};
    const abby::vec2 size{93.2f, 933.3f};
    tree.emplace(2, position, size);
    CHECK(tree.size() == 2);

    const auto& aabb = tree.get_aabb(2);
    CHECK(aabb.min == position);
    CHECK(aabb.max == position + size);
  }

  TEST_CASE("aabb_tree::erase")
  {
    abby::aabb_tree<int> tree;
    CHECK_NOTHROW(tree.erase(-1));
    CHECK_NOTHROW(tree.erase(0));
    CHECK_NOTHROW(tree.erase(1));

    tree.insert(4, abby::make_aabb<float>({0, 0}, {10, 10}));
    tree.insert(7, abby::make_aabb<float>({120, 33}, {50, 50}));
    CHECK(tree.size() == 2);

    tree.erase(4);
    CHECK(tree.size() == 1);

    CHECK_NOTHROW(tree.erase(4));
    CHECK(tree.size() == 1);

    tree.erase(7);
    CHECK(tree.size() == 0);
    CHECK(tree.is_empty());
  }

  TEST_CASE("aabb_tree::replace")
  {
    abby::aabb_tree<int> tree;
    CHECK_NOTHROW(tree.replace(0, {}));

    tree.insert(35, abby::make_aabb<float>({34, 63}, {31, 950}));
    tree.insert(99, abby::make_aabb<float>({2, 412}, {78, 34}));

    const auto id = 27;
    const auto original = abby::make_aabb<float>({0, 0}, {100, 100});
    tree.insert(id, original);

    SUBCASE("Update to smaller AABB")
    {
      // When the new AABB is smaller, nothing is done
      tree.replace(id, abby::make_aabb<float>({10, 10}, {50, 50}));

      const auto& actual = tree.get_aabb(id);
      CHECK(original.min == actual.min);
      CHECK(original.max == actual.max);
      CHECK(original.area() == actual.area());
    }

    SUBCASE("Update to larger AABB")
    {
      const auto large = abby::make_aabb<float>({20, 20}, {150, 150});
      tree.replace(id, large);

      const auto& actual = tree.get_aabb(id);
      CHECK(large.min == actual.min);
      CHECK(large.max == actual.max);
      CHECK(large.area() == actual.area());
    }
  }

  TEST_CASE("aabb_tree::relocate")
  {
    abby::aabb_tree<int> tree;
    CHECK_NOTHROW(tree.relocate(0, {}));

    tree.insert(7, abby::make_aabb<float>({12, 34}, {56, 78}));
    tree.insert(2, {{1, 2}, {3, 4}});
    tree.insert(84, {{91, 22}, {422, 938}});

    const abby::vec2<float> pos{389, 534};
    tree.relocate(7, pos);

    CHECK(tree.size() == 3);
    CHECK(tree.get_aabb(7).min == pos);
  }

  TEST_CASE("aabb_tree::query_collisions")
  {
    SUBCASE("Empty tree")
    {
      abby::aabb_tree<int> tree;
      std::vector<int> candidates;

      CHECK_NOTHROW(tree.query_collisions(0, std::back_inserter(candidates)));
      CHECK(candidates.empty());
    }

    SUBCASE("Populated tree")
    {
      abby::aabb_tree<int> tree;
      std::vector<int> candidates;

      tree.insert(1, abby::make_aabb<float>({10, 10}, {100, 100}));
      tree.insert(2, abby::make_aabb<float>({90, 10}, {50, 50}));
      tree.insert(3, abby::make_aabb<float>({10, 90}, {25, 25}));

      tree.query_collisions(1, std::back_inserter(candidates));
      CHECK(candidates.size() == 2);
      CHECK_FALSE(
          std::any_of(begin(candidates), end(candidates), [](auto candidate) {
            return candidate == 1;
          }));
      CHECK(std::any_of(begin(candidates), end(candidates), [](auto candidate) {
        return candidate == 2;
      }));
      CHECK(std::any_of(begin(candidates), end(candidates), [](auto candidate) {
        return candidate == 3;
      }));
    }
  }

  TEST_CASE("aabb_tree::get_aabb")
  {
    abby::aabb_tree<int> tree;
    CHECK_THROWS(tree.get_aabb(0));

    const auto aabb = abby::make_aabb<float>({12, 34}, {56, 78});
    tree.insert(12, aabb);
    CHECK(tree.get_aabb(12) == aabb);
  }

  TEST_CASE("aabb_tree::size")
  {
    abby::aabb_tree<int> tree;
    CHECK(tree.size() == 0);

    tree.insert(0, {});
    CHECK(tree.size() == 1);

    tree.insert(1, {});
    CHECK(tree.size() == 2);

    tree.erase(1);
    CHECK(tree.size() == 1);

    tree.erase(0);
    CHECK(tree.size() == 0);
  }

  TEST_CASE("aabb_tree::is_empty")
  {
    abby::aabb_tree<int> tree;
    CHECK(tree.is_empty());

    tree.insert(123, {});
    CHECK(!tree.is_empty());

    tree.erase(123);
    CHECK(tree.is_empty());
  }
}
