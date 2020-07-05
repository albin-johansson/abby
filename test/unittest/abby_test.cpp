#include "abby.hpp"

#include <doctest.h>

using namespace abby;

namespace {

void validate_aabb(const aabb& current, const aabb& original)
{
  CHECK(current.min.x == original.min.x);
  CHECK(current.min.y == original.min.y);

  CHECK(current.max.x == original.max.x);
  CHECK(current.max.y == original.max.y);

  CHECK(current.area == original.area);

  CHECK(current.center.x == original.center.x);
  CHECK(current.center.y == original.center.y);
}

}  // namespace

TEST_SUITE("abby")
{
  TEST_CASE("abby::insert")
  {
    entt::registry registry;
    const entt::entity null{entt::null};

    const auto fstID = registry.create();
    const auto fstBox = make_aabb({10, 10}, {100, 100});

    insert(registry, fstID, fstBox);

    SUBCASE("State after adding one AABB")
    {
      CHECK(!registry.empty<detail::root>());
      CHECK(registry.size<detail::node>() == 1);

      const auto& node = registry.get<detail::node>(fstID);

      CHECK(node.parent == null);
      CHECK(node.leftChild == null);
      CHECK(node.rightChild == null);

      validate_aabb(node.aabb, fstBox);
    }

    const auto sndID = registry.create();
    const auto sndBox = make_aabb({110, 110}, {80, 80});

    insert(registry, sndID, sndBox);

    SUBCASE("State after adding two AABBs")
    {
      CHECK(!registry.empty<detail::root>());
      CHECK(registry.size<detail::node>() == 3);

      const auto& node = registry.get<detail::node>(sndID);

      CHECK(node.parent == null);
      CHECK(node.leftChild == null);
      CHECK(node.rightChild == null);

      validate_aabb(node.aabb, sndBox);
    }

    SUBCASE("Root after two AABBs added")
    {
      const auto root = registry.view<detail::root>().front();
      const auto& rootNode = registry.get<detail::node>(root);

      CHECK(rootNode.parent == null);
      CHECK(rootNode.leftChild != null);
      CHECK(rootNode.rightChild != null);
    }
    // TODO test with more AABBs...
  }
}