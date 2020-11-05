#include <doctest.h>

#include "abby.hpp"

TEST_SUITE("aabb")
{
  TEST_CASE("Default values")
  {
    const abby::faabb aabb;
    CHECK(aabb.min == abby::fvec2{0, 0});
    CHECK(aabb.max == abby::fvec2{0, 0});
  }

  TEST_CASE("aabb::operator==")
  {
    SUBCASE("Self")
    {
      const abby::faabb aabb;
      CHECK(aabb == aabb);
    }

    SUBCASE("Two equal AABBs")
    {
      const abby::faabb fst{{10, 20}, {65, 82}};
      const abby::faabb snd{fst};
      CHECK(fst == snd);
      CHECK(snd == fst);
    }

    SUBCASE("Two different AABBs")
    {
      const abby::faabb fst{{73, 12}, {341, 275}};
      const abby::faabb snd{{27, 63}, {299, 512}};
      CHECK_FALSE(fst == snd);
      CHECK_FALSE(snd == fst);
    }
  }

  TEST_CASE("aabb::operator!=")
  {
    SUBCASE("Self")
    {
      const abby::faabb aabb;
      CHECK_FALSE(aabb != aabb);
    }

    SUBCASE("Two equal AABBs")
    {
      const abby::faabb fst{{45, 66}, {346, 992}};
      const abby::faabb snd{fst};
      CHECK_FALSE(fst != snd);
      CHECK_FALSE(snd != fst);
    }

    SUBCASE("Two different AABBs")
    {
      const abby::faabb fst{{55, 76}, {476, 775}};
      const abby::faabb snd{{29, 44}, {345, 173}};
      CHECK(fst != snd);
      CHECK(snd != fst);
    }
  }

  TEST_CASE("make_aabb")
  {
    const abby::fvec2 pos{27, 93};
    const abby::fvec2 size{871, 712};
    const auto aabb = abby::make_aabb(pos, size);

    CHECK(aabb.min == pos);
    CHECK(aabb.max - aabb.min == size);
  }

  TEST_CASE("aabb::area")
  {
    SUBCASE("Empty")
    {
      const abby::faabb empty;
      CHECK(empty.area() == 0);
    }

    SUBCASE("Non-empty")
    {
      const auto aabb = abby::make_aabb<float>({10, 25}, {100, 75});
      const auto area = aabb.area();

      const auto diff = aabb.max - aabb.min;
      CHECK(area == diff.x * diff.y);
    }
  }

  TEST_CASE("combine")
  {
    const auto fst = abby::make_aabb<float>({10, 15}, {200, 250});
    const auto snd = abby::make_aabb<float>({83, 64}, {155, 62});
    const auto combined = abby::combine(fst, snd);

    CHECK(combined.min.x == std::min(fst.min.x, snd.min.x));
    CHECK(combined.min.y == std::min(fst.min.y, snd.min.y));
    CHECK(combined.max.x == std::max(fst.max.x, snd.max.x));
    CHECK(combined.max.y == std::max(fst.max.y, snd.max.y));
  }

  TEST_CASE("aabb::contains")
  {
    SUBCASE("Self")
    {
      const auto aabb = abby::make_aabb<float>({0, 0}, {10, 10});
      CHECK(aabb.contains(aabb));
    }

    SUBCASE("Aligned borders")
    {
      const auto fst = abby::make_aabb<float>({0, 0}, {10, 10});
      const auto snd = abby::make_aabb<float>({5, 5}, {5, 5});
      CHECK(fst.contains(snd));
    }

    SUBCASE("1 pixel outside")
    {
      const auto fst = abby::make_aabb<float>({0, 0}, {10, 10});
      const auto snd = abby::make_aabb<float>({6, 6}, {5, 5});
      CHECK(!fst.contains(snd));
    }

    SUBCASE("1 pixel margin")
    {
      const auto fst = abby::make_aabb<float>({0, 0}, {10, 10});
      const auto snd = abby::make_aabb<float>({4, 4}, {5, 5});
      CHECK(fst.contains(snd));
    }
  }

  TEST_CASE("aabb::overlaps")
  {
    SUBCASE("Self")
    {
      const auto aabb = abby::make_aabb<float>({0, 0}, {10, 10});
      CHECK(aabb.overlaps(aabb));
    }

    SUBCASE("Aligned borders")
    {
      const auto fst = abby::make_aabb<float>({0, 0}, {10, 10});
      const auto snd = abby::make_aabb<float>({5, 5}, {5, 5});
      CHECK(fst.overlaps(snd));
    }

    SUBCASE("1 pixel outside")
    {
      const auto fst = abby::make_aabb<float>({0, 0}, {10, 10});
      const auto snd = abby::make_aabb<float>({6, 6}, {5, 5});
      CHECK(fst.overlaps(snd));
    }

    SUBCASE("1 pixel margin")
    {
      const auto fst = abby::make_aabb<float>({0, 0}, {10, 10});
      const auto snd = abby::make_aabb<float>({4, 4}, {5, 5});
      CHECK(fst.overlaps(snd));
    }
  }
}
