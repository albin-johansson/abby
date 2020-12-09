#include <AABB.h>
#include <doctest.h>

#include <iterator>

#include "abby.hpp"

namespace {

struct dummy_vector final
{
  using value_type = double;
  value_type x{};
  value_type y{};

  [[nodiscard]] auto operator+(const dummy_vector& other) const noexcept
      -> dummy_vector
  {
    return {x + other.x, y + other.y};
  }

  [[nodiscard]] auto operator-(const dummy_vector& other) const noexcept
      -> dummy_vector
  {
    return {x - other.x, y - other.y};
  }
};

using tree_t = abby::tree<int>;
using aabb_t = tree_t::aabb_type;

}  // namespace

TEST_SUITE("tree")
{
  TEST_CASE("tree::insert")
  {
    tree_t tree;
    REQUIRE(tree.is_empty());

    tree.insert(1, {0, 0}, {100, 100});
    CHECK(!tree.is_empty());
    CHECK(tree.size() == 1);

    tree.insert(2, {40, 40}, {100, 100});
    CHECK(tree.size() == 2);

    tree.insert(3, {75, 75}, {100, 100});
    CHECK(tree.size() == 3);
  }

  TEST_CASE("tree with custom vector")
  {
    abby::tree<int, double, dummy_vector> tree;
    tree.set_thickness_factor(std::nullopt);

    const dummy_vector min{345, 122};
    const dummy_vector max{564, 931};

    tree.insert(7, min, max);

    const auto& aabb = tree.get_aabb(7);
    CHECK(aabb.min().x == min.x);
    CHECK(aabb.min().y == min.y);
    CHECK(aabb.max().x == max.x);
    CHECK(aabb.max().y == max.y);
  }

  TEST_CASE("tree::erase")
  {
    tree_t tree;

    CHECK_NOTHROW(tree.erase(-1));
    CHECK_NOTHROW(tree.erase(0));
    CHECK_NOTHROW(tree.erase(1));

    tree.insert(4, {0, 0}, {10, 10});
    tree.insert(7, {120, 33}, {170, 93});
    CHECK(tree.size() == 2);

    tree.erase(4);
    CHECK(tree.size() == 1);

    CHECK_NOTHROW(tree.erase(4));
    CHECK(tree.size() == 1);

    tree.erase(7);
    CHECK(tree.size() == 0);
    CHECK(tree.is_empty());
  }

  TEST_CASE("tree::update")
  {
    tree_t tree;
    tree.set_thickness_factor(std::nullopt);

    CHECK_NOTHROW(tree.update(0, {{0, 0}, {10, 10}}));

    tree.insert(35, {34, 63}, {65, 950});
    tree.insert(99, {2, 34}, {78, 412});

    const auto id = 27;
    const tree_t::aabb_type original{{0, 0}, {100, 100}};
    tree.insert(id, original.min(), original.max());

    SUBCASE("Update to smaller AABB")
    {
      tree.update(id, {10, 10}, {60, 60});

      {
        const auto& aabb = tree.get_aabb(id);
        CHECK(original.min() == aabb.min());
        CHECK(original.max() == aabb.max());
        CHECK(original.area() == aabb.area());
      }

      tree.update(id, {12, 13}, {65, 74}, true);
      const auto& aabb = tree.get_aabb(id);
      CHECK(aabb.min().x == 12);
      CHECK(aabb.min().y == 13);
      CHECK(aabb.max().x == 65);
      CHECK(aabb.max().y == 74);
    }

    SUBCASE("Update to larger AABB")
    {
      const aabb_t large{{20, 20}, {170, 170}};
      tree.update(id, large);

      const auto& actual = tree.get_aabb(id);
      CHECK(large.min() == actual.min());
      CHECK(large.max() == actual.max());
      CHECK(large.area() == actual.area());
    }
  }

  TEST_CASE("tree::relocate")
  {
    tree_t tree;
    tree.set_thickness_factor(std::nullopt);

    CHECK_NOTHROW(tree.relocate(0, {}));

    tree.insert(7, {12, 34}, {68, 112});
    tree.insert(2, {1, 2}, {4, 6});
    tree.insert(84, {91, 22}, {422, 938});

    const abby::vector2<double> pos{389, 534};
    tree.relocate(7, pos);

    CHECK(tree.size() == 3);
    CHECK(tree.get_aabb(7).min() == pos);
  }

  TEST_CASE("tree::rebuild")
  {
    tree_t tree;
    CHECK_NOTHROW(tree.rebuild());

    tree.insert(1, {10, 10}, {20, 20});
    tree.insert(2, {54, 76}, {243, 123});
    tree.insert(3, {88, 43}, {674, 786});
    tree.insert(4, {342, 124}, {7753, 2412});
    tree.insert(5, {56, 352}, {1245, 3576});

    CHECK_NOTHROW(tree.rebuild());
  }

  TEST_CASE("tree::query")
  {
    SUBCASE("Empty tree")
    {
      tree_t tree;
      std::vector<int> candidates;

      CHECK_NOTHROW(tree.query(0, std::back_inserter(candidates)));
      CHECK(candidates.empty());
    }

    SUBCASE("Populated tree")
    {
      tree_t tree;
      std::vector<int> candidates;

      tree.insert(1, {10, 10}, {110, 110});
      tree.insert(2, {90, 10}, {160, 60});
      tree.insert(3, {10, 90}, {35, 115});

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

  TEST_CASE("tree::query_direct")
  {
    SUBCASE("Empty tree")
    {
      tree_t tree;
      CHECK_NOTHROW(tree.query_direct(0, [](int i) {
        DOCTEST_FAIL("Should not get any matches!");
      }));
    }

    SUBCASE("Non-empty tree")
    {
      tree_t tree;
      CHECK_NOTHROW(tree.query_direct(1, [](int) {
        DOCTEST_FAIL("Should have no matches!");
      }));

      tree.insert(1, {10, 10}, {110, 110});
      tree.insert(2, {90, 10}, {160, 60});
      tree.insert(3, {10, 90}, {35, 115});

      int count{0};
      tree.query_direct(1, [&](int candidate) {
        ++count;
      });
      CHECK(count == 2);

      count = 0;
      tree.query_direct(1, [&](int candidate) {
        ++count;
        return false;
      });
      CHECK(count == 2);

      count = 0;
      tree.query_direct(1, [&](int candidate) {
        ++count;
        return true;
      });
      CHECK(count == 1);
    }
  }

  TEST_CASE("tree::get_aabb")
  {
    tree_t tree;
    tree.set_thickness_factor(std::nullopt);

    CHECK_THROWS(tree.get_aabb(0));

    const aabb_t aabb{{12, 34}, {56, 78}};
    tree.insert(12, aabb.min(), aabb.max());
    CHECK(tree.get_aabb(12) == aabb);
  }

  TEST_CASE("tree::height")
  {
    tree_t tree;
    CHECK(tree.height() == 0);

    tree.insert(1, {45, 23}, {66, 236});
    CHECK(tree.height() == 0);

    tree.insert(2, {22, 12}, {73, 91});
    CHECK(tree.height() == 1);

    tree.insert(3, {123, 321}, {456, 654});
    CHECK(tree.height() == 2);
  }

  TEST_CASE("tree::size")
  {
    tree_t tree;
    CHECK(tree.size() == 0);

    tree.insert(0, {}, {1, 1});
    CHECK(tree.size() == 1);

    tree.insert(1, {}, {1, 1});
    CHECK(tree.size() == 2);

    tree.erase(1);
    CHECK(tree.size() == 1);

    tree.erase(0);
    CHECK(tree.size() == 0);
  }

  TEST_CASE("tree::node_count")
  {
    tree_t tree;
    CHECK(tree.node_count() == 0);

    tree.insert(1, {10, 10}, {20, 20});
    CHECK(tree.node_count() == 1);

    tree.insert(2, {40, 40}, {60, 55});
    CHECK(tree.node_count() == 3);

    tree.insert(3, {121, 145}, {432, 234});
    CHECK(tree.node_count() == 5);

    tree.erase(2);
    CHECK(tree.node_count() == 3);

    tree.erase(3);
    CHECK(tree.node_count() == 1);
  }

  TEST_CASE("tree::is_empty")
  {
    tree_t tree;
    CHECK(tree.is_empty());

    tree.insert(123, {}, {1, 1});
    CHECK(!tree.is_empty());

    tree.erase(123);
    CHECK(tree.is_empty());
  }

  TEST_CASE("tree::thickness_factor")
  {
    tree_t tree;
    CHECK(tree.thickness_factor() == 0.05);

    tree.set_thickness_factor(0.5);
    CHECK(tree.thickness_factor() == 0.5);

    tree.set_thickness_factor(std::nullopt);
    CHECK(!tree.thickness_factor());
  }

  TEST_CASE("tree with many AABBs")
  {
    tree_t tree{24};
    tree.set_thickness_factor(std::nullopt);

    tree.insert(1, {182, 831}, {416, 1770});
    tree.insert(2, {3845, 31}, {3901, 54});
    tree.insert(3, {6752, 3411}, {7517, 3665});
    tree.insert(4, {675, 883}, {906, 970});
    tree.insert(5, {468, 454}, {5039, 2796});
    tree.insert(6, {334, 1091}, {411, 1815});
    tree.insert(7, {786, 1234}, {45357, 1278});
    tree.insert(8, {12313, 4333}, {13100, 4789});
    tree.insert(9, {12, 767}, {357, 811});
    tree.insert(10, {995, 34}, {1560, 48});
    tree.insert(11, {213, 4230}, {1090, 4386});
    tree.insert(12, {4665, 125}, {4699, 1360});
    tree.insert(13, {7381, 5783}, {7513, 7017});
    tree.print(std::clog);

    std::vector<int> vec;
    tree.query(1, std::back_inserter(vec));

    tree.update(6, {145, 321}, {311, 752});

    tree.erase(7);
    tree.erase(5);
    tree.erase(11);
    tree.erase(2);
    tree.erase(13);
    tree.erase(9);

    tree.clear();

    CHECK_NOTHROW(tree.relocate(1, {}));
  }
}
