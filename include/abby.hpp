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
 */

#include <algorithm>        // min, max
#include <array>            // array
#include <cassert>          // assert
#include <cstddef>          // byte
#include <deque>            // deque
#include <map>              // map
#include <memory_resource>  // monotonic_buffer_resource
#include <optional>         // optional
#include <stack>            // stack
#include <vector>           // vector

#ifndef ABBY_HEADER_GUARD
#define ABBY_HEADER_GUARD

namespace abby {

using opt_int = std::optional<int>;

template <typename T>
struct vec2 final
{
  T x{};
  T y{};
};

template <typename T = float>
struct aabb final
{
  vec2<T> min;
  vec2<T> max;
};

template <typename T>
[[nodiscard]] auto area_of(const aabb<T>& aabb) noexcept -> T
{
  const auto width = aabb.max.x - aabb.min.x;
  const auto height = aabb.max.y - aabb.min.y;
  return width * height;
}

template <typename T>
[[nodiscard]] auto combine(const aabb<T>& fst, const aabb<T>& snd) noexcept
    -> aabb<T>
{
  aabb result;

  result.min.x = std::min(fst.min.x, snd.min.x);
  result.min.y = std::min(fst.min.y, snd.min.y);

  result.max.x = std::max(fst.max.x, snd.max.x);
  result.max.y = std::max(fst.max.y, snd.max.y);

  return result;
}

template <typename T>
[[nodiscard]] auto contains(const aabb<T>& source,
                            const aabb<T>& other) noexcept -> bool
{
  return other.min.x() >= source.min.x() && other.max.x() <= source.max.x() &&
         other.min.y() >= source.min.y() && other.max.y() <= source.max.y();
}

template <typename T>
[[nodiscard]] auto overlaps(const aabb<T>& fst, const aabb<T>& snd) noexcept
    -> bool
{
  return (fst.max.x() > snd.min.x()) && (fst.min.x() < snd.max.x()) &&
         (fst.max.y() > snd.min.y()) && (fst.min.y() < snd.max.y());
}

/**
 * \struct aabb_node
 *
 * \brief Represents a node in an AABB tree.
 *
 * \details Contains an AABB and the entity associated with the AABB, along
 * with tree information.
 *
 * \headerfile abby.hpp
 */
template <typename T, typename U = float>
struct aabb_node final
{
  T id;
  aabb<U> box;
  opt_int parent;
  opt_int left;
  opt_int right;
  opt_int next;
};

template <typename T, typename U>
[[nodiscard]] inline auto is_leaf(const aabb_node<T, U>& node) noexcept -> bool
{
  return !node.left;
}

template <typename T, typename U>
[[nodiscard]] auto get_left_cost(const aabb_node<T, U>& left,
                                 const aabb_node<T, U>& leaf,
                                 float minimumCost) -> float
{
  if (is_leaf(left)) {
    return area_of(combine(leaf.box, left.box)) + minimumCost;
  } else {
    const auto newLeftAabb = combine(leaf.box, left.box);
    return (area_of(newLeftAabb) - area_of(left.box)) + minimumCost;
  }
}

template <typename T, typename U>
[[nodiscard]] auto get_right_cost(const aabb_node<T, U>& right,
                                  const aabb_node<T, U>& leaf,
                                  float minimumCost) -> float
{
  if (is_leaf(right)) {
    return area_of(combine(leaf.box, right.box)) + minimumCost;
  } else {
    const auto newRightAabb = combine(leaf.box, right.box);
    return (area_of(newRightAabb) - area_of(right.box)) + minimumCost;
  }
}

template <typename Key, typename T = float>
class aabb_tree final
{
  template <typename U>
  using pmr_stack = std::stack<U, std::pmr::deque<U>>;

 public:
  using key_type = Key;
  using size_type = std::size_t;
  using vector_type = vec2<T>;
  using aabb_type = aabb<T>;
  using node_type = aabb_node<key_type, T>;
  using index_type = int;

  aabb_tree()
  {
    m_nodes.reserve(m_nodeCapacity);
    m_allocatedNodes = m_nodeCapacity;

    const auto size = static_cast<int>(m_nodeCapacity);
    for (int index = 0; index < size; ++index) {
      auto& node = m_nodes.emplace_back();
      node.next = index + 1;
    }

    m_nodes.at(size - 1).next.reset();
  }

  void insert(const key_type& key, const aabb_type& box)
  {
    const auto index = allocate_node();
    auto& node = m_nodes.at(index);
    node.box = box;
    node.id = key;

    insert_leaf(index);
    m_indexMap.emplace(key, index);
  }

  void remove(const key_type& key)
  {
    const auto index = m_indexMap.at(key);
    remove_leaf(index);
    deallocate_node(index);
    m_indexMap.erase(key);
  }

  void update(const key_type& key, const aabb_type& box)
  {
    update_leaf(m_indexMap.at(key), box);
  }

  void set_position(const key_type& key, const vector_type& position)
  {
    const auto previous = get_aabb(key);

    aabb newBox;
    newBox.min = position;
    newBox.max = position + (previous.max - previous.min);

    //    const auto width = newBox.max.x() - newBox.min.x();
    //    const auto height = newBox.max.y() - newBox.min.y();
    //    newBox.area = width * height;

    update(key, newBox);
  }

  template <typename OutputIterator>
  void query_collisions(const key_type& key, OutputIterator iterator) const
  {
    std::array<std::byte, sizeof(opt_int) * 32> buffer;  // NOLINT
    std::pmr::monotonic_buffer_resource resource{buffer.data(), sizeof buffer};
    pmr_stack<std::optional<int>> stack{&resource};

    const auto& box = get_aabb(key);

    stack.push(m_rootIndex);
    while (!stack.empty()) {
      const auto nodeIndex = stack.top();
      stack.pop();

      if (!nodeIndex.has_value()) {
        continue;
      }

      const auto& node = m_nodes.at(*nodeIndex);
      if (overlaps(node.box, box)) {
        if (is_leaf(node) && node.entity != key) {
          *iterator = node.entity;
          ++iterator;
        } else {
          stack.push(node.left);
          stack.push(node.right);
        }
      }
    }
  }

  [[nodiscard]] auto get_aabb(const key_type& key) const -> const aabb_type&
  {
    return m_nodes.at(m_indexMap.at(key)).box;
  }

  [[nodiscard]] auto size() const noexcept -> size_type
  {
    return m_indexMap.size();
  }

  [[nodiscard]] auto is_empty() const noexcept -> bool
  {
    return m_indexMap.empty();
  }

 private:
  using opt_index = std::optional<index_type>;

  std::map<key_type, index_type> m_indexMap;
  std::vector<aabb_node<key_type, T>> m_nodes;

  opt_index m_rootIndex{};
  opt_index m_nextFreeNodeIndex{};

  size_type m_allocatedNodes{};
  size_type m_nodeCapacity{24};
  size_type m_growthSize{m_nodeCapacity};

  void grow_pool()
  {
    assert(m_allocatedNodes == m_nodeCapacity);

    m_nodeCapacity += m_growthSize;
    m_nodes.resize(m_nodeCapacity);

    for (index_type index = m_allocatedNodes; index < m_nodeCapacity; ++index) {
      auto& node = m_nodes.at(index);
      node.next = index + 1;
    }

    m_nodes.at(m_nodeCapacity - 1).next.reset();
    m_nextFreeNodeIndex = static_cast<index_type>(m_allocatedNodes);
  }

  [[nodiscard]] auto allocate_node() -> index_type
  {
    // if we have no free tree nodes then grow the pool
    if (!m_nextFreeNodeIndex.has_value()) {
      grow_pool();
    }

    const auto index = m_nextFreeNodeIndex.value();
    const auto& node = m_nodes.at(index);
    m_nextFreeNodeIndex = node.next;
    ++m_allocatedNodes;

    return index;
  }

  void deallocate_node(index_type nodeIndex)
  {
    auto& node = m_nodes.at(nodeIndex);
    node.next = m_nextFreeNodeIndex;
    m_nextFreeNodeIndex = nodeIndex;
    --m_allocatedNodes;
  }

  void fix_upwards_tree(opt_index nodeIndex)
  {
    while (nodeIndex) {
      auto& node = m_nodes.at(nodeIndex.value());

      // every node should be a parent
      assert(node.left);
      assert(node.right);

      // fix height and area
      const auto& left = m_nodes.at(node.left.value());
      const auto& right = m_nodes.at(node.right.value());
      node.box = combine(left.box, right.box);

      nodeIndex = node.parent;
    }
  }

  [[nodiscard]] auto find_best_insertion_position(
      const node_type& leafNode) const -> index_type
  {
    auto treeNodeIndex = m_rootIndex.value();
    while (!is_leaf(m_nodes.at(treeNodeIndex))) {
      // because of the test in the while loop above we know we are never a leaf
      // inside it
      const auto& treeNode = m_nodes.at(treeNodeIndex);
      const auto leftNodeIndex = treeNode.left.value();
      const auto rightNodeIndex = treeNode.right.value();
      const auto& leftNode = m_nodes.at(leftNodeIndex);
      const auto& rightNode = m_nodes.at(rightNodeIndex);

      const auto combined = combine(treeNode.box, leafNode.box);

      const auto newParentNodeCost = 2.0f * area_of(combined);
      const auto minimumPushDownCost =
          2.0f * (area_of(combined) - area_of(treeNode.box));

      // use the costs to figure out whether to create a new parent here or
      // descend
      const auto costLeft =
          get_left_cost(leftNode, leafNode, minimumPushDownCost);
      const auto costRight =
          get_right_cost(rightNode, leafNode, minimumPushDownCost);

      // if the cost of creating a new parent node here is less than descending
      // in either direction then we know we need to create a new parent node
      // here and attach the leaf to that
      if (newParentNodeCost < costLeft && newParentNodeCost < costRight) {
        break;
      }

      // otherwise descend in the cheapest direction
      if (costLeft < costRight) {
        treeNodeIndex = leftNodeIndex;
      } else {
        treeNodeIndex = rightNodeIndex;
      }
    }

    return treeNodeIndex;
  }

  void insert_leaf(index_type leafIndex)
  {
    // make sure we're inserting a new leaf
    assert(!m_nodes.at(leafIndex).parent);
    assert(!m_nodes.at(leafIndex).left);
    assert(!m_nodes.at(leafIndex).right);

    // if the tree is empty then we make the root the leaf
    if (!m_rootIndex) {
      m_rootIndex = leafIndex;
      return;
    }

    // search for the best place to put the new leaf in the tree
    // we use surface area and depth as search heuristics
    auto& leafNode = m_nodes.at(leafIndex);
    const auto foundIndex = find_best_insertion_position(leafNode);

    // the leafs sibling is going to be the node we found above and we are going
    // to create a new parent node and attach the leaf and this item
    const auto leafSiblingIndex = foundIndex;
    auto& leafSibling = m_nodes.at(leafSiblingIndex);

    const auto oldParentIndex = leafSibling.parent;
    const auto newParentIndex = allocate_node();

    auto& newParent = m_nodes.at(newParentIndex);
    newParent.parent = oldParentIndex;
    // the new parents aabb is the leaf aabb combined with it's siblings aabb
    newParent.box = combine(leafNode.box, leafSibling.box);
    newParent.left = leafSiblingIndex;
    newParent.right = leafIndex;
    leafNode.parent = newParentIndex;
    leafSibling.parent = newParentIndex;

    if (!oldParentIndex.has_value()) {
      // the old parent was the root and so this is now the root
      m_rootIndex = newParentIndex;
    } else {
      // the old parent was not the root and so we need to patch the left or
      // right index to point to the new node
      auto& oldParent = m_nodes.at(oldParentIndex.value());
      if (oldParent.left == leafSiblingIndex) {
        oldParent.left = newParentIndex;
      } else {
        oldParent.right = newParentIndex;
      }
    }

    // finally we need to walk back up the tree fixing heights and areas
    fix_upwards_tree(leafNode.parent.value());
  }

  void remove_leaf(index_type leafIndex)
  {
    // if the leaf is the root then we can just clear the root pointer and
    // return
    if (leafIndex == m_rootIndex) {
      m_rootIndex = std::nullopt;
      return;
    }

    auto& leafNode = m_nodes.at(leafIndex);

    const auto parentNodeIndex = leafNode.parent.value();
    const auto& parentNode = m_nodes.at(parentNodeIndex);

    const auto grandParentNodeIndex = parentNode.parent;

    const auto siblingNodeIndex =
        parentNode.left == leafIndex ? parentNode.right : parentNode.left;
    assert(siblingNodeIndex.has_value());  // we must have a sibling
    auto& siblingNode = m_nodes.at(*siblingNodeIndex);

    if (grandParentNodeIndex.has_value()) {
      // if we have a grand parent (i.e. the parent is not the root) then
      // destroy the parent and connect the sibling to the grandparent in its
      // place
      auto& grandParentNode = m_nodes.at(*grandParentNodeIndex);
      if (grandParentNode.left == parentNodeIndex) {
        grandParentNode.left = siblingNodeIndex;
      } else {
        grandParentNode.right = siblingNodeIndex;
      }
      siblingNode.parent = grandParentNodeIndex;
      deallocate_node(parentNodeIndex);
      fix_upwards_tree(grandParentNodeIndex);
    } else {
      // if we have no grandparent then the parent is the root and so our
      // sibling becomes the root and has it's parent removed
      m_rootIndex = siblingNodeIndex;
      siblingNode.parent = std::nullopt;
      deallocate_node(parentNodeIndex);
    }

    leafNode.parent = std::nullopt;
  }

  void update_leaf(index_type leafIndex, const aabb_type& box)
  {
    auto& node = m_nodes.at(leafIndex);

    // if the node contains the new aabb then we just leave things
    if (contains(node.box, box)) {
      return;
    }

    remove_leaf(leafIndex);
    node.box = box;
    insert_leaf(leafIndex);
  }
};

}  // namespace abby

#endif  // ABBY_HEADER_GUARD
