#include "abby.hpp"

#include <doctest.h>

TEST_SUITE("abby")
{
  TEST_CASE("abby::insert")
  {
    entt::registry registry;
    const auto id = registry.create();

    abby::aabb box;

    abby::insert(registry, id, box);
    abby::remove(registry, id);
    abby::remove_all(registry);
    abby::update(registry, id);
    abby::update(registry, id, true);
    const auto hits = abby::query(registry, id);
    abby::query(registry, id, [](const abby::aabb& box) {

    });
    const auto size = abby::size(registry);
  }
}