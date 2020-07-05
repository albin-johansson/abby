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

auto get_node(entt::registry& registry, const entt::entity nodeEntity)
-> detail::node&
{
  return registry.get<detail::node>(nodeEntity);
}

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
  const auto& node = get_node(registry, nodeEntity);
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

void set_root(entt::registry& registry, const entt::entity nodeEntity)
{
  registry.clear<detail::root>();
  registry.emplace<detail::root>(nodeEntity);
}

[[nodiscard]] auto balance(entt::registry& registry,
                           const entt::entity nodeEntity) -> entt::entity
{
  auto& node = get_node(registry, nodeEntity);

  if (is_leaf(node) || node.height < 2) {
    return nodeEntity;
  }

  const auto left = node.leftChild;
  const auto right = node.rightChild;

  auto& leftNode = get_node(registry, left);
  auto& rightNode = get_node(registry, right);

  const auto currentBalance = rightNode.height - leftNode.height;

  // Rotate right branch up.
  if (currentBalance > 1) {
    const auto rightLeft = rightNode.leftChild;
    const auto rightRight = rightNode.rightChild;

    // Swap node and its right-hand child.
    rightNode.leftChild = nodeEntity;
    rightNode.parent = node.parent;
    node.parent = right;

    // The node's old parent should now point to its right-hand child.
    if (rightNode.parent != entt::null) {
      auto& rightParentNode = get_node(registry, rightNode.parent);
      if (rightParentNode.leftChild == nodeEntity) {
        rightParentNode.leftChild = right;
      } else {
        BOOST_ASSERT(rightParentNode.rightChild == nodeEntity);
        rightParentNode.rightChild = right;
      }
    } else {
      registry.clear<detail::root>();
      registry.emplace<detail::root>(right);
    }

    // Rotate.
    auto& rightLeftNode = get_node(registry, rightLeft);
    auto& rightRightNode = get_node(registry, rightRight);
    if (rightLeftNode.height > rightRightNode.height) {
      rightNode.rightChild = rightLeft;
      node.rightChild = rightRight;
      rightRightNode.parent = nodeEntity;

      node.aabb = detail::merge(leftNode.aabb, rightRightNode.aabb);
      rightNode.aabb = detail::merge(node.aabb, rightLeftNode.aabb);

      node.height = 1 + std::max(leftNode.height, rightRightNode.height);
      rightNode.height = 1 + std::max(node.height, rightLeftNode.height);
    } else {
      rightNode.rightChild = rightRight;
      node.rightChild = rightLeft;
      rightLeftNode.parent = nodeEntity;

      node.aabb = detail::merge(leftNode.aabb, rightLeftNode.aabb);
      rightNode.aabb = detail::merge(node.aabb, rightRightNode.aabb);

      node.height = 1 + std::max(leftNode.height, rightLeftNode.height);
      rightNode.height = 1 + std::max(node.height, rightRightNode.height);
    }

    return right;
  }

  // Rotate left branch up.
  if (currentBalance < -1) {
    const auto leftLeft = leftNode.leftChild;
    const auto leftRight = leftNode.rightChild;

    // Swap node and its left-hand child.
    leftNode.leftChild = nodeEntity;
    leftNode.parent = node.parent;
    node.parent = left;

    // The node's old parent should now point to its left-hand child.
    if (leftNode.parent != entt::null) {
      auto& leftParentNode = get_node(registry, leftNode.parent);
      if (leftParentNode.leftChild == nodeEntity) {
        leftParentNode.leftChild = left;
      } else {
        BOOST_ASSERT(leftParentNode.rightChild == nodeEntity);
        leftParentNode.rightChild = left;
      }
    } else {
      set_root(registry, left);
    }

    // Rotate.
    auto& leftLeftNode = get_node(registry, leftLeft);
    auto& leftRightNode = get_node(registry, leftRight);
    if (leftLeftNode.height > leftRightNode.height) {
      leftNode.rightChild = leftLeft;
      node.leftChild = leftRight;
      leftRightNode.parent = nodeEntity;

      node.aabb = detail::merge(rightNode.aabb, leftRightNode.aabb);
      leftNode.aabb = detail::merge(node.aabb, leftLeftNode.aabb);

      node.height = 1 + std::max(rightNode.height, leftRightNode.height);
      leftNode.height = 1 + std::max(node.height, leftLeftNode.height);
    } else {
      leftNode.rightChild = leftRight;
      node.leftChild = leftLeft;
      leftLeftNode.parent = nodeEntity;

      node.aabb = detail::merge(rightNode.aabb, leftLeftNode.aabb);
      leftNode.aabb = detail::merge(node.aabb, leftRightNode.aabb);

      node.height = 1 + std::max(rightNode.height, leftLeftNode.height);
      leftNode.height = 1 + std::max(node.height, leftRightNode.height);
    }

    return left;
  }

  return nodeEntity;
}

auto find_best_sibling(entt::registry& registry, const aabb& leafAABB)
    -> entt::entity
{
  auto id = registry.view<detail::root>().front();
  while (!detail::is_leaf(registry, id)) {
    // Extract the children of the node.
    const auto& node = get_node(registry, id);
    const auto left = node.leftChild;
    const auto right = node.rightChild;

    const auto combinedAABB = detail::merge(node.aabb, leafAABB);

    const auto combinedArea = detail::area(combinedAABB);

    // Cost of creating a new parent for this node and the new leaf.
    const auto cost = 2.0f * combinedArea;

    // Minimum cost of pushing the leaf further down the tree.
    const auto inheritanceCost = 2.0f * (combinedArea - node.aabb.area);

    const auto& leftNode = get_node(registry, left);
    const auto& rightNode = get_node(registry, right);

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
  return id;
}

void fix_tree_upwards(entt::registry& registry, entt::entity id)
{
  // Walk back up the tree fixing heights and AABBs.
  while (id != entt::null) {
    id = balance(registry, id);

    auto& node = get_node(registry, id);

    const auto left = node.leftChild;
    const auto right = node.rightChild;

    BOOST_ASSERT(left != entt::null);
    BOOST_ASSERT(right != entt::null);

    const auto& leftNode = get_node(registry, left);
    const auto& rightNode = get_node(registry, right);

    node.height = 1 + std::max(leftNode.height, rightNode.height);
    node.aabb = detail::merge(leftNode.aabb, rightNode.aabb);

    id = node.parent;
  }
}

void insert_leaf(entt::registry& registry, const entt::entity leaf)
{
  if (registry.empty<detail::root>()) {
    registry.emplace<detail::root>(leaf);
    auto& leafNode = get_node(registry, leaf);
    leafNode.parent = entt::null;
    return;
  }

  // Find the best sibling for the node.
  auto& leafNode = get_node(registry, leaf);
  const auto& leafAABB = leafNode.aabb;

  const auto sibling = find_best_sibling(registry, leafAABB);
  auto& siblingNode = get_node(registry, sibling);

  // Create a new parent.
  const auto oldParent = siblingNode.parent;
  const auto newParent = registry.create();

  auto& newParentNode = registry.emplace<detail::node>(newParent);
  newParentNode.parent = oldParent;
  newParentNode.aabb = detail::merge(leafAABB, siblingNode.aabb);
  newParentNode.height = siblingNode.height + 1;

  if (oldParent != entt::null) {
    auto& oldParentNode = get_node(registry, oldParent);
    if (oldParentNode.leftChild == sibling) {
      oldParentNode.leftChild = newParent;
    } else {
      oldParentNode.rightChild = newParent;
    }
  } else {
    // The sibling was the root.
    set_root(registry, newParent);
  }

  newParentNode.leftChild = sibling;
  newParentNode.rightChild = leaf;
  siblingNode.parent = newParent;
  leafNode.parent = newParent;

  fix_tree_upwards(registry, leafNode.parent);
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
