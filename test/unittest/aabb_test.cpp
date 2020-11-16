#include <AABB.h>
#include <doctest.h>

#include "abby.hpp"

using vec2 = abby::vector2<double>;
using aabb_t = abby::aabb<double>;

namespace {

struct dummy_vector final
{
  double x{};
  double y{};
};

}  // namespace

TEST_SUITE("aabb")
{
  TEST_CASE("Default values")
  {
    const aabb_t aabb;
    CHECK(aabb.min() == vec2{0, 0});
    CHECK(aabb.max() == vec2{0, 0});
  }

  TEST_CASE("aabb::vector2 ctor")
  {
    const vec2 min{12, 34};
    const vec2 max{56, 78};

    const aabb_t aabb{min, max};

    CHECK(aabb.min().x == min.x);
    CHECK(aabb.min().y == min.y);
    CHECK(aabb.max().x == max.x);
    CHECK(aabb.max().y == max.y);
  }

  TEST_CASE("aabb::templated ctor")
  {
    const dummy_vector min{12, 34};
    const dummy_vector max{56, 78};

    const aabb_t aabb{min, max};

    CHECK(aabb.min().x == min.x);
    CHECK(aabb.min().y == min.y);
    CHECK(aabb.max().x == max.x);
    CHECK(aabb.max().y == max.y);
  }

  TEST_CASE("aabb::fatten")
  {
    SUBCASE("Fatten AABB")
    {
      aabb_t aabb{{12, 24}, {47, 54}};

      const auto areaBefore = aabb.area();
      aabb.fatten(0.05);

      const auto areaAfter = aabb.area();
      CHECK(areaAfter > areaBefore);
    }

    SUBCASE("No-op")
    {
      aabb_t aabb{{12, 24}, {47, 54}};

      const auto areaBefore = aabb.area();
      aabb.fatten(std::nullopt);

      const auto areaAfter = aabb.area();
      CHECK(areaBefore == areaAfter);
    }
  }

  TEST_CASE("aabb::area")
  {
    SUBCASE("Empty")
    {
      const aabb_t empty;
      CHECK(empty.area() == 0);
    }

    SUBCASE("Non-empty")
    {
      const aabb_t abby{{0, 0}, {10, 10}};
      const aabb::AABB aabb{{0, 0}, {10, 10}};

      CHECK(abby.area() == aabb.getSurfaceArea());
      CHECK(abby.compute_area() == aabb.computeSurfaceArea());
    }
  }

  TEST_CASE("aabb::size")
  {
    const auto width = 123;
    const auto height = 345;

    const vec2 pos{45, 32};
    const aabb_t aabb{pos, pos + vec2{width, height}};

    CHECK(aabb.size().x == width);
    CHECK(aabb.size().y == height);
  }

  TEST_CASE("aabb::min")
  {
    const vec2 min{231, 453};
    const aabb_t aabb{min, {999, 999}};
    CHECK(aabb.min() == min);
  }

  TEST_CASE("aabb::max")
  {
    const vec2 max{786, 448};
    const aabb_t aabb{{111, 111}, max};
    CHECK(aabb.max() == max);
  }

  TEST_CASE("aabb::merge")
  {
    const aabb_t fst{{10, 15}, {200, 250}};
    const aabb_t snd{{83, 64}, {155, 126}};
    const auto combined = aabb_t::merge(fst, snd);

    CHECK(combined.min().x == std::min(fst.min().x, snd.min().x));
    CHECK(combined.min().y == std::min(fst.min().y, snd.min().y));
    CHECK(combined.max().x == std::max(fst.max().x, snd.max().x));
    CHECK(combined.max().y == std::max(fst.max().y, snd.max().y));
  }

  TEST_CASE("aabb::contains")
  {
    SUBCASE("Self")
    {
      const aabb_t aabb{{0, 0}, {10, 10}};
      CHECK(aabb.contains(aabb));
    }

    SUBCASE("Aligned borders")
    {
      const aabb_t fst{{0, 0}, {10, 10}};
      const aabb_t snd{{5, 5}, {10, 10}};
      CHECK(fst.contains(snd));
    }

    SUBCASE("1 pixel outside")
    {
      const aabb_t fst{{0, 0}, {10, 10}};
      const aabb_t snd{{6, 6}, {11, 11}};
      CHECK(!fst.contains(snd));
    }

    SUBCASE("1 pixel margin")
    {
      const aabb_t fst{{0, 0}, {10, 10}};
      const aabb_t snd{{4, 4}, {9, 9}};
      CHECK(fst.contains(snd));
    }
  }

  TEST_CASE("aabb::overlaps")
  {
    SUBCASE("Self")
    {
      const aabb_t aabb{{0, 0}, {10, 10}};
      CHECK(aabb.overlaps(aabb, true));
    }

    SUBCASE("Aligned borders")
    {
      const aabb_t fst{{0, 0}, {10, 10}};
      const aabb_t snd{{5, 5}, {10, 10}};
      CHECK(fst.overlaps(snd, true));
    }

    SUBCASE("1 pixel outside")
    {
      const aabb_t fst{{0, 0}, {10, 10}};
      const aabb_t snd{{6, 6}, {11, 11}};
      CHECK(fst.overlaps(snd, true));
    }

    SUBCASE("1 pixel margin")
    {
      const aabb_t fst{{0, 0}, {10, 10}};
      const aabb_t snd{{4, 4}, {9, 9}};
      CHECK(fst.overlaps(snd, true));
    }
  }

  TEST_CASE("aabb::operator==")
  {
    SUBCASE("Self")
    {
      const aabb_t aabb;
      CHECK(aabb == aabb);
    }

    SUBCASE("Two equal AABBs")
    {
      const aabb_t fst{{10, 20}, {65, 82}};
      const aabb_t snd{fst};
      CHECK(fst == snd);
      CHECK(snd == fst);
    }

    SUBCASE("Two different AABBs")
    {
      const aabb_t fst{{73, 12}, {341, 275}};
      const aabb_t snd{{27, 63}, {299, 512}};
      CHECK_FALSE(fst == snd);
      CHECK_FALSE(snd == fst);
    }
  }

  TEST_CASE("aabb::operator!=")
  {
    SUBCASE("Self")
    {
      const aabb_t aabb;
      CHECK_FALSE(aabb != aabb);
    }

    SUBCASE("Two equal AABBs")
    {
      const aabb_t fst{{45, 66}, {346, 992}};
      const aabb_t snd{fst};
      CHECK_FALSE(fst != snd);
      CHECK_FALSE(snd != fst);
    }

    SUBCASE("Two different AABBs")
    {
      const aabb_t fst{{55, 76}, {476, 775}};
      const aabb_t snd{{29, 44}, {345, 173}};
      CHECK(fst != snd);
      CHECK(snd != fst);
    }
  }
}
