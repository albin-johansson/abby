/*
 * MIT License
 *
 * Copyright (c) 2020 Albin Johansson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * This code was adapted from the AABB.cc library, which can be found here:
 * https://github.com/lohedges/aabbcc. The AABB.cc library is licensed under
 * the Zlib license.
 */

#ifndef ABBY_HEADER_GUARD
#define ABBY_HEADER_GUARD

#include <assert.hpp>
#include <entt.hpp>

/**
 * @namespace abby
 * @brief The top-level namespace that contains the components of the library.
 */
namespace abby {

/**
 * @struct vec2
 *
 * @brief Represents a simple floating-point 2-dimensional vector.
 *
 * @since 0.1.0
 *
 * @var vec2::x
 * The x-coordinate of the vector.
 *
 * @var vec2::y
 * The y-coordinate of the vector.
 *
 * @headerfile abby.hpp
 */
struct vec2 final {
  float x{};
  float y{};
};

/**
 * @struct aabb
 *
 * @brief Represents an AABB (Axis Aligned Bounding Box).
 *
 * @details An AABB is really just a fancy rectangle. They are mainly used
 * for collision detection systems, where trees can be built using AABBs in
 * order to decrease the complexity of finding potential collision candidates.
 * However, AABBs are not used for detailed collision detection. They are only
 * used to find potential collisions, which are then checked using more exact
 * collision detection systems.
 *
 * @note The "axis aligned" part is important, it means that the axes of the
 * rectangles must be aligned (parallel in relation to each other).
 * Otherwise, the systems that rely on AABBs won't work.
 *
 * @since 0.1.0
 *
 * @var aabb::min
 * The minimum x- and y-coordinates. Which are the coordinates of the
 * north-west corner of the box.
 *
 * @var aabb::max
 * The maximum x- and y-coordinates. Which are the coordinates of the
 * south-east corner of the box.
 *
 * @var aabb::center
 * The coordinates of the center point of the box.
 *
 * @var aabb::area
 * The area of the box.
 *
 * @headerfile abby.hpp
 */
struct aabb {
  vec2 min;
  vec2 max;
  vec2 center;
  float area{};
};

//! @cond FALSE

namespace detail {

struct root final {
};

struct node final {
  abby::aabb aabb;
  entt::entity parent{entt::null};
  entt::entity leftChild{entt::null};
  entt::entity rightChild{entt::null};
  int height{};
};

[[nodiscard]] auto center(const aabb& box) noexcept -> vec2
{
  const auto width = box.max.x - box.min.x;
  const auto height = box.max.y - box.min.y;
  return {width / 2.0f, height / 2.0f};
};

[[nodiscard]] auto area(const aabb& box) noexcept -> float
{
  const auto width = box.max.x - box.min.x;
  const auto height = box.max.y - box.min.y;
  return width * height;
};

[[nodiscard]] auto is_leaf(const detail::node& node) noexcept -> bool
{
  return node.leftChild == entt::null;
}

[[nodiscard]] auto is_leaf(entt::registry& registry,
                           const entt::entity nodeEntity) noexcept -> bool
{
  const auto& node = registry.get<detail::node>(nodeEntity);
  return is_leaf(node);
}

[[nodiscard]] auto merge(const aabb& fst, const aabb& snd) noexcept -> aabb
{
  aabb result;

  result.min.x = std::min(fst.min.x, snd.min.x);
  result.min.y = std::min(fst.min.y, snd.min.y);

  result.max.x = std::max(fst.max.x, snd.max.x);
  result.max.y = std::max(fst.max.y, snd.max.y);

  result.area = area(result);
  result.center = center(result);

  return result;
};

[[nodiscard]] auto balance(entt::registry& registry,
                           const entt::entity nodeEntity) -> entt::entity
{
  // TODO implement
  return nodeEntity;
}

void insert_leaf(entt::registry& registry, const entt::entity leaf)
{
  if (registry.empty<detail::root>()) {
    registry.emplace<detail::root>(leaf);
    auto& leafNode = registry.get<detail::node>(leaf);
    leafNode.parent = entt::null;
    return;
  }

  // Find the best sibling for the node.
  auto& leafNode = registry.get<detail::node>(leaf);
  const auto& leafAABB = leafNode.aabb;

  auto id = registry.view<detail::root>().front();
  while (!detail::is_leaf(registry, id)) {
    // Extract the children of the node.
    const auto& node = registry.get<detail::node>(id);
    const auto left = node.leftChild;
    const auto right = node.rightChild;

    const auto combinedAABB = detail::merge(node.aabb, leafAABB);

    const auto combinedArea = detail::area(combinedAABB);

    // Cost of creating a new parent for this node and the new leaf.
    const auto cost = 2.0f * combinedArea;

    // Minimum cost of pushing the leaf further down the tree.
    const auto inheritanceCost = 2.0f * (combinedArea - node.aabb.area);

    const auto& leftNode = registry.get<detail::node>(left);
    const auto& rightNode = registry.get<detail::node>(right);

    const auto getCost = [&](const detail::node& node) noexcept -> float {
      const auto box = detail::merge(leafAABB, node.aabb);
      if (detail::is_leaf(node)) {
        return box.area + inheritanceCost;
      } else {
        const auto oldArea = leftNode.aabb.area;
        const auto newArea = box.area;
        return (newArea - oldArea) + inheritanceCost;
      }
    };

    const auto costLeft = getCost(leftNode);
    const auto costRight = getCost(rightNode);

    // Descend according to the minimum cost.
    if ((cost < costLeft) && (cost < costRight)) {
      break;
    } else {
      if (costLeft < costRight) {
        id = left;
      } else {
        id = right;
      }
    }
  }

  const auto sibling = id;
  auto& siblingNode = registry.get<detail::node>(sibling);

  // Create a new parent.
  const auto oldParent = siblingNode.parent;
  const auto newParent = registry.create();

  auto& newParentNode = registry.emplace<detail::node>(newParent);
  newParentNode.parent = oldParent;
  newParentNode.aabb = detail::merge(leafAABB, siblingNode.aabb);
  newParentNode.height = siblingNode.height + 1;

  if (oldParent != entt::null) {
    auto& oldParentNode = registry.get<detail::node>(oldParent);
    if (oldParentNode.leftChild == sibling) {
      oldParentNode.leftChild = newParent;
    } else {
      oldParentNode.rightChild = newParent;
    }
  } else {
    // The sibling was the root.
    registry.clear<detail::root>();
    registry.emplace<detail::root>(newParent);
  }

  newParentNode.leftChild = sibling;
  newParentNode.rightChild = leaf;
  siblingNode.parent = newParent;
  leafNode.parent = newParent;

  // Walk back up the tree fixing heights and AABBs.
  id = leafNode.parent;
  while (id != entt::null) {
    id = balance(registry, id);

    auto& node = registry.get<detail::node>(id);

    const auto left = node.leftChild;
    const auto right = node.rightChild;

    BOOST_ASSERT(left != entt::null);
    BOOST_ASSERT(right != entt::null);

    const auto& leftNode = registry.get<detail::node>(left);
    const auto& rightNode = registry.get<detail::node>(right);

    node.height = 1 + std::max(leftNode.height, rightNode.height);
    node.aabb = detail::merge(leftNode.aabb, rightNode.aabb);

    id = node.parent;
  }
}

}  // namespace detail

//! @endcond

[[nodiscard]] auto make_aabb(const vec2& position, const vec2& size) noexcept
    -> aabb;

void insert(entt::registry& registry, entt::entity id, const aabb& box);

}  // namespace abby

auto abby::make_aabb(const abby::vec2& position,
                     const abby::vec2& size) noexcept -> abby::aabb
{
  aabb result;

  result.min.x = position.x;
  result.min.y = position.y;

  result.max.x = position.x + size.x;
  result.max.y = position.y + size.y;

  result.area = detail::area(result);
  result.center = detail::center(result);

  return result;
}

void abby::insert(entt::registry& registry,
                  entt::entity id,
                  const abby::aabb& box)
{
  BOOST_ASSERT_MSG(!registry.has<detail::node>(id),
                   "Entity already associated with node!");

  // TODO fatten the AABB?

  auto& node = registry.emplace<detail::node>(id);
  node.aabb = box;
  node.aabb.center = detail::center(box);
  node.aabb.area = detail::area(box);
  node.height = 0;

  detail::insert_leaf(registry, id);
}

#endif  // ABBY_HEADER_GUARD
