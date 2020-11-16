#include <AABB.h>
#include <doctest.h>

#include "abby.hpp"

using vec2 = abby::vector2<double>;
using aabb_t = abby::aabb<double>;

TEST_SUITE("aabb")
{
  TEST_CASE("Default values")
  {
    const aabb_t aabb;
    CHECK(aabb.min() == vec2{0, 0});
    CHECK(aabb.max() == vec2{0, 0});
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

  TEST_CASE("merge")
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
}
