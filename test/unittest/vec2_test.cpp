#include <doctest.h>

#include "abby.hpp"

TEST_SUITE("vec2")
{
  TEST_CASE("vec2 default values")
  {
    const abby::fvec2 vec;
    CHECK(vec.x == 0);
    CHECK(vec.y == 0);
  }

  TEST_CASE("vec2::operator+")
  {
    const abby::fvec2 fst{776, 141};
    const abby::fvec2 snd{514, 482};
    const auto sum = fst + snd;
    CHECK(sum.x == fst.x + snd.x);
    CHECK(sum.y == fst.y + snd.y);
  }

  TEST_CASE("vec2::operator-")
  {
    const abby::fvec2 fst{912, 56};
    const abby::fvec2 snd{448, -167};

    const auto diff1 = fst - snd;
    CHECK(diff1.x == fst.x - snd.x);
    CHECK(diff1.y == fst.y - snd.y);

    const auto diff2 = snd - fst;
    CHECK(diff2.x == snd.x - fst.x);
    CHECK(diff2.y == snd.y - fst.y);
  }

  TEST_CASE("vec2::operator==")
  {
    SUBCASE("Self")
    {
      const abby::fvec2 vec;
      CHECK(vec == vec);
    }

    SUBCASE("Two equivalent vectors")
    {
      const abby::fvec2 fst{24, 48};
      const abby::fvec2 snd{fst};
      CHECK(fst == snd);
      CHECK(snd == fst);
    }

    SUBCASE("Two different vectors")
    {
      const abby::fvec2 fst{24, 48};
      const abby::fvec2 snd{83, 123};
      CHECK_FALSE(fst == snd);
      CHECK_FALSE(snd == fst);
    }
  }

  TEST_CASE("vec2::operator!=")
  {
    SUBCASE("Self")
    {
      const abby::fvec2 vec;
      CHECK_FALSE(vec != vec);
    }

    SUBCASE("Two equivalent vectors")
    {
      const abby::fvec2 fst{456, 284};
      const abby::fvec2 snd{fst};
      CHECK_FALSE(fst != snd);
      CHECK_FALSE(snd != fst);
    }

    SUBCASE("Two different vectors")
    {
      const abby::fvec2 fst{82, 76};
      const abby::fvec2 snd{148, 897};
      CHECK(fst != snd);
      CHECK(snd != fst);
    }
  }
}
