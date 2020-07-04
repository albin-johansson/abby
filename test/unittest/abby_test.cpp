#include "abby.hpp"

#include <doctest.h>

using namespace abby;

TEST_SUITE("abby")
{
  TEST_CASE("abby::insert")
  {
    entt::registry registry;

    const auto fstID = registry.create();
    const auto fstBox = make_aabb({10, 10}, {100, 100});

    insert(registry, fstID, fstBox);

    SUBCASE("State after adding one AABB")
    {
      CHECK(!registry.empty<detail::root>());
      CHECK(registry.size<detail::node>() == 1);

      const auto& node = registry.get<detail::node>(fstID);
      const entt::entity null{entt::null};

      CHECK(node.parent == null);
      CHECK(node.leftChild == null);
      CHECK(node.rightChild == null);
      CHECK(node.height == 0);

      const auto& aabb = node.aabb;
      CHECK(aabb.min.x == fstBox.min.x);
      CHECK(aabb.min.y == fstBox.min.y);

      CHECK(aabb.max.x == fstBox.max.x);
      CHECK(aabb.max.y == fstBox.max.y);

      CHECK(aabb.area == fstBox.area);

      CHECK(aabb.center.x == fstBox.center.x);
      CHECK(aabb.center.y == fstBox.center.y);
    }

    // TODO test with more AABBs...
  }
}