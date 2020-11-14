#include <doctest.h>

#include <iterator>

#include "abby.hpp"

TEST_SUITE("tree")
{
  TEST_CASE("tree::insert")
  {
    abby::tree<int> tree;
    REQUIRE(tree.is_empty());

    tree.insert(1, abby::make_aabb<float>({0, 0}, {100, 100}));
    CHECK(!tree.is_empty());
    CHECK(tree.size() == 1);

    tree.insert(2, abby::make_aabb<float>({40, 40}, {100, 100}));
    CHECK(tree.size() == 2);

    tree.insert(3, abby::make_aabb<float>({75, 75}, {100, 100}));
    CHECK(tree.size() == 3);
  }

  TEST_CASE("tree::emplace")
  {
    abby::tree<int> tree;
    tree.set_fattening_factor(std::nullopt);

    tree.emplace(1, abby::vec2{1.0f, 2.0f}, abby::vec2{10.0f, 12.0f});
    CHECK(!tree.is_empty());
    CHECK(tree.size() == 1);

    const auto& fst = tree.get_aabb(1);
    CHECK(fst.min.x == 1.0f);
    CHECK(fst.min.y == 2.0f);
    CHECK(fst.max.x == 11.0f);
    CHECK(fst.max.y == 14.0f);

    const abby::vec2 position{89.3f, 123.4f};
    const abby::vec2 size{93.2f, 933.3f};
    tree.emplace(2, position, size);
    CHECK(tree.size() == 2);

    const auto& snd = tree.get_aabb(2);
    CHECK(snd.min == position);
    CHECK(snd.max == position + size);
  }

  TEST_CASE("tree::erase")
  {
    abby::tree<int> tree;
    tree.set_fattening_factor(std::nullopt);

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

  TEST_CASE("tree::replace")
  {
    abby::tree<int> tree;
    tree.set_fattening_factor(std::nullopt);

    CHECK_NOTHROW(tree.replace(0, {}));

    tree.insert(35, abby::make_aabb<float>({34, 63}, {31, 950}));
    tree.insert(99, abby::make_aabb<float>({2, 412}, {78, 34}));

    const auto id = 27;
    const auto original = abby::make_aabb<float>({0, 0}, {100, 100});
    tree.insert(id, original);

    SUBCASE("Update to smaller AABB")
    {
      tree.replace(id, abby::make_aabb<float>({10, 10}, {50, 50}));
      {
        const auto& aabb = tree.get_aabb(id);
        CHECK(original.min == aabb.min);
        CHECK(original.max == aabb.max);
        CHECK(original.area() == aabb.area());
      }

      tree.replace(id, abby::make_aabb<float>({10, 10}, {50, 50}), true);
      const auto& aabb = tree.get_aabb(id);
      CHECK(aabb.min.x == 10);
      CHECK(aabb.min.y == 10);
      CHECK(aabb.max.y == 60);
      CHECK(aabb.max.y == 60);
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

  TEST_CASE("tree::relocate")
  {
    abby::tree<int> tree;
    tree.set_fattening_factor(std::nullopt);

    CHECK_NOTHROW(tree.relocate(0, {}));

    tree.insert(7, abby::make_aabb<float>({12, 34}, {56, 78}));
    tree.insert(2, {{1, 2}, {3, 4}});
    tree.insert(84, {{91, 22}, {422, 938}});

    const abby::vec2<float> pos{389, 534};
    tree.relocate(7, pos);

    CHECK(tree.size() == 3);
    CHECK(tree.get_aabb(7).min == pos);
  }

  TEST_CASE("tree::query")
  {
    SUBCASE("Empty tree")
    {
      abby::tree<int> tree;
      std::vector<int> candidates;

      CHECK_NOTHROW(tree.query(0, std::back_inserter(candidates)));
      CHECK(candidates.empty());
    }

    SUBCASE("Populated tree")
    {
      abby::tree<int> tree;
      std::vector<int> candidates;

      tree.insert(1, abby::make_aabb<float>({10, 10}, {100, 100}));
      tree.insert(2, abby::make_aabb<float>({90, 10}, {50, 50}));
      tree.insert(3, abby::make_aabb<float>({10, 90}, {25, 25}));

      tree.query(1, std::back_inserter(candidates));
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

  TEST_CASE("tree::get_aabb")
  {
    abby::tree<int> tree;
    tree.set_fattening_factor(std::nullopt);

    CHECK_THROWS(tree.get_aabb(0));

    const auto aabb = abby::make_aabb<float>({12, 34}, {56, 78});
    tree.insert(12, aabb);
    CHECK(tree.get_aabb(12) == aabb);
  }

  TEST_CASE("tree::size")
  {
    abby::tree<int> tree;
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

  TEST_CASE("tree::is_empty")
  {
    abby::tree<int> tree;
    CHECK(tree.is_empty());

    tree.insert(123, {});
    CHECK(!tree.is_empty());

    tree.erase(123);
    CHECK(tree.is_empty());
  }

  TEST_CASE("tree with many AABBs") {
    abby::tree<int> tree;

    tree.emplace(1, {182, 831}, {234, 939});
    tree.emplace(2, {3845, 31}, {56, 23});
    tree.emplace(3, {6752, 3411}, {765, 254});
    tree.emplace(4, {675, 883}, {231, 87});
    tree.emplace(5, {468, 454}, {4571, 2342});
    tree.emplace(6, {334, 1091}, {77, 724});
    tree.emplace(7, {786, 1234}, {44571, 44});
    tree.emplace(8, {12313, 4333}, {787, 456});
    tree.emplace(9, {12, 767}, {345, 44});
    tree.emplace(10, {995, 34}, {565, 14});
    tree.emplace(11, {213, 4230}, {877, 156});
    tree.emplace(12, {4665, 125}, {34, 1235});
    tree.emplace(13, {7381, 5783}, {132, 1234});
  }
}
