#pragma once

#include <algorithm>      // min, max
#include <cassert>        // assert
#include <cmath>          // abs
#include <limits>         // numeric_limits
#include <optional>       // optional
#include <ostream>        // ostream
#include <stack>          // stack
#include <stdexcept>      // invalid_argument
#include <string>         // string
#include <unordered_map>  // unordered_map
#include <vector>         // vector

namespace abby2 {
const unsigned int NULL_NODE = 0xffffffff;

template <typename T>
struct vector2 final
{
  T x{};
  T y{};

  [[deprecated]] auto operator[](std::size_t index) -> T&
  {
    if (index == 0) {
      return x;
    } else if (index == 1) {
      return y;
    } else {
      throw std::invalid_argument{"vector2: bad subscript index!"};
    }
  }

  [[deprecated]] auto operator[](std::size_t index) const -> const T&
  {
    if (index == 0) {
      return x;
    } else if (index == 1) {
      return y;
    } else {
      throw std::invalid_argument{"vector2: bad subscript index!"};
    }
  }
};

template <typename T>
class aabb final
{
 public:
  using vector_type = vector2<T>;

  constexpr aabb() noexcept = default;

  constexpr aabb(const vector_type& min, const vector_type& max)
      : m_min{min},
        m_max{max},
        m_area{compute_area()}
  {
    if ((m_min.x > m_max.x) || (m_min.y > m_max.y)) {
      throw std::invalid_argument("AABB: min > max");
    }
  }

  constexpr void merge(const aabb& fst, const aabb& snd)
  {
    //    m_min.x = std::min(fst.m_min.x, snd.m_min.x);
    //    m_min.y = std::min(fst.m_min.y, snd.m_min.y);
    //
    //    m_max.x = std::min(fst.m_max.x, snd.m_max.x);
    //    m_max.y = std::min(fst.m_max.y, snd.m_max.y);
    for (auto i = 0; i < 2; i++) {
      m_min[i] = std::min(fst.m_min[i], snd.m_min[i]);
      m_max[i] = std::max(fst.m_max[i], snd.m_max[i]);
    }

    m_area = compute_area();
    //    centre = computeCentre();
  }

  [[nodiscard]] constexpr auto contains(const aabb& other) const noexcept
      -> bool
  {
    //    if ((other.m_min.x < m_min.x) || (other.m_max.x > m_max.x) ||
    //        (other.m_min.y < m_min.y) || (other.m_max.y > m_max.y)) {
    //      return false;
    //    } else {
    //      return true;
    //    }
    for (unsigned int i = 0; i < 2; i++) {
      if (other.m_min[i] < m_min[i]) return false;
      if (other.m_max[i] > m_max[i]) return false;
    }
    return true;
  }

  [[nodiscard]] constexpr auto overlaps(const aabb& other,
                                        bool touchIsOverlap) const noexcept
      -> bool
  {
    //    if (touchIsOverlap) {
    //      if ((other.m_max.x < m_min.x || other.m_min.x > m_max.x) ||
    //          (other.m_max.y < m_min.y || other.m_min.y > m_max.y)) {
    //        return false;
    //      }
    //    } else {
    //      if ((other.m_max.x <= m_min.x || other.m_min.x >= m_max.x) ||
    //          (other.m_max.y <= m_min.y || other.m_min.y >= m_max.y)) {
    //        return false;
    //      }
    //    }
    //    return true;
    bool rv = true;

    if (touchIsOverlap) {
      for (unsigned int i = 0; i < 2; ++i) {
        if (other.m_max[i] < m_min[i] || other.m_min[i] > m_max[i]) {
          rv = false;
          break;
        }
      }
    } else {
      for (unsigned int i = 0; i < 2; ++i) {
        if (other.m_max[i] <= m_min[i] || other.m_min[i] >= m_max[i]) {
          rv = false;
          break;
        }
      }
    }

    return rv;
  }

  [[nodiscard]] constexpr auto compute_area() const noexcept -> double
  {
    //    const auto width = m_max.x - m_min.x;
    //    const auto height = m_max.y - m_max.y;
    //    return width * height;
    // Sum of "area" of all the sides.
    double sum = 0;

    // General formula for one side: hold one dimension constant
    // and multiply by all the other ones.
    for (unsigned int d1 = 0; d1 < 2; d1++) {
      // "Area" of current side.
      double product = 1;

      for (unsigned int d2 = 0; d2 < 2; d2++) {
        if (d1 == d2) continue;

        double dx = m_max[d2] - m_min[d2];
        product *= dx;
      }

      // Update the sum.
      sum += product;
    }

    return 2.0 * sum;
  }

  [[nodiscard]] constexpr auto area() const noexcept -> double
  {
    return m_area;
  }

  // private:
  vector_type m_min;
  vector_type m_max;
  double m_area{};
};

template <typename Key, typename T>
struct node final
{
  // TODO optional indices and ID

  using key_type = Key;
  using aabb_type = aabb<T>;
  using index_type = unsigned int;

  key_type id;
  aabb_type aabb;

  index_type parent;
  index_type next;
  index_type left;
  index_type right;
  int height;

  [[nodiscard]] auto is_leaf() const noexcept -> bool
  {
    return (left == NULL_NODE);
  }
};

template <typename Key, typename T = double>
class tree final
{
 public:
  using value_type = T;
  using key_type = Key;
  using vector_type = vector2<value_type>;
  using aabb_type = aabb<value_type>;
  using node_type = node<key_type, value_type>;
  using size_type = std::size_t;
  using index_type = unsigned int;
  using maybe_index = std::optional<index_type>;

  [[deprecated]] explicit tree(double skinThickness = 0.05,
                               size_type nParticles = 16,
                               bool touchIsOverlap = true)
      : m_skinThickness{skinThickness},
        m_touchIsOverlap{touchIsOverlap}
  {
    // Initialise the tree.
    m_root = NULL_NODE;
    m_nodeCount = 0;
    m_nodeCapacity = nParticles;
    m_nodes.resize(m_nodeCapacity);

    // Build a linked list for the list of free nodes.
    for (auto i = 0; i < m_nodeCapacity - 1; i++) {
      m_nodes[i].next = i + 1;
      m_nodes[i].height = -1;
    }
    m_nodes[m_nodeCapacity - 1].next = NULL_NODE;
    m_nodes[m_nodeCapacity - 1].height = -1;

    // Assign the index of the first free node.
    m_nextFreeIndex = 0;
  }

  void insert(const key_type& id,
              vector_type& lowerBound,
              vector_type& upperBound)
  {
    // Make sure the particle doesn't already exist.
    if (m_indexMap.count(id) != 0) {
      throw std::invalid_argument("[ERROR]: Particle already exists in tree!");
    }

    // Allocate a new node for the particle.
    unsigned int node = allocateNode();

    // AABB size in each dimension.
    vector_type size;

    // Compute the AABB limits.
    for (unsigned int i = 0; i < 2; i++) {
      // Validate the bound.
      if (lowerBound[i] > upperBound[i]) {
        throw std::invalid_argument(
            "[ERROR]: AABB lower bound is greater than the upper bound!");
      }

      m_nodes[node].aabb.m_min[i] = lowerBound[i];
      m_nodes[node].aabb.m_max[i] = upperBound[i];
      size[i] = upperBound[i] - lowerBound[i];
    }

    // Fatten the AABB.
    for (unsigned int i = 0; i < 2; i++) {
      m_nodes[node].aabb.m_min[i] -= m_skinThickness * size[i];
      m_nodes[node].aabb.m_max[i] += m_skinThickness * size[i];
    }
    m_nodes[node].aabb.m_area = m_nodes[node].aabb.compute_area();
    //    m_nodes[node].aabb.m_centre = m_nodes[node].aabb.computeCentre();

    // Zero the height.
    m_nodes[node].height = 0;

    // Insert a new leaf into the tree.
    insertLeaf(node);

    // Add the new particle to the map.
    m_indexMap.insert(
        std::unordered_map<unsigned int, unsigned int>::value_type(id, node));

    // Store the particle index.
    m_nodes[node].id = id;

#ifndef NDEBUG
    validate();
#endif
  }

  void erase(const key_type& id)
  {
    // Map iterator.
    std::unordered_map<unsigned int, unsigned int>::iterator it;

    // Find the particle.
    it = m_indexMap.find(id);

    // The particle doesn't exist.
    if (it == m_indexMap.end()) {
      throw std::invalid_argument("[ERROR]: Invalid particle index!");
    }

    // Extract the node index.
    unsigned int node = it->second;

    // Erase the particle from the map.
    m_indexMap.erase(it);

    assert(node < m_nodeCapacity);
    assert(m_nodes[node].is_leaf());

    removeLeaf(node);
    freeNode(node);

#ifndef NDEBUG
    validate();
#endif
  }

  void clear()
  {
    // Iterator pointing to the start of the particle map.
    std::unordered_map<unsigned int, unsigned int>::iterator it =
        m_indexMap.begin();

    // Iterate over the map.
    while (it != m_indexMap.end()) {
      // Extract the node index.
      unsigned int node = it->second;

      assert(node < m_nodeCapacity);
      assert(m_nodes[node].is_leaf());

      removeLeaf(node);
      freeNode(node);

      it++;
    }

    // Clear the particle map.
    m_indexMap.clear();

#ifndef NDEBUG
    validate();
#endif
  }

  void print(std::ostream& stream) const
  {
    stream << "abby2:\n";
    print(stream, "", m_root, false);
  }

  void print(std::ostream& stream,
             const std::string& prefix,
             index_type index,
             bool isLeft) const
  {
    if (index != NULL_NODE) {
      const auto& node = m_nodes.at(index);

      stream << prefix << (isLeft ? "├── " : "└── ");
      if (node.is_leaf()) {
        stream << node.id << "\n";
      } else {
        stream << "X\n";
      }

      print(stream, prefix + (isLeft ? "│   " : "    "), node.left, true);
      print(stream, prefix + (isLeft ? "│   " : "    "), node.right, false);
    }
  }

  auto update(const key_type& id,
              vector_type& lowerBound,
              vector_type& upperBound,
              bool alwaysReinsert = false) -> bool
  {
    // Validate the dimensionality of the bounds vectors.
    if ((lowerBound.size() != dimension) && (upperBound.size() != dimension)) {
      throw std::invalid_argument("[ERROR]: Dimensionality mismatch!");
    }

    // Map iterator.
    std::unordered_map<unsigned int, unsigned int>::iterator it;

    // Find the particle.
    it = m_indexMap.find(id);

    // The particle doesn't exist.
    if (it == m_indexMap.end()) {
      throw std::invalid_argument("[ERROR]: Invalid particle index!");
    }

    // Extract the node index.
    unsigned int node = it->second;

    assert(node < m_nodeCapacity);
    assert(m_nodes[node].is_leaf());

    // AABB size in each dimension.
    std::vector<double> size(dimension);

    // Compute the AABB limits.
    for (unsigned int i = 0; i < dimension; i++) {
      // Validate the bound.
      if (lowerBound[i] > upperBound[i]) {
        throw std::invalid_argument(
            "[ERROR]: AABB lower bound is greater than the upper bound!");
      }

      size[i] = upperBound[i] - lowerBound[i];
    }

    // Create the new AABB.
    aabb_type aabb(lowerBound, upperBound);

    // No need to update if the particle is still within its fattened AABB.
    if (!alwaysReinsert && m_nodes[node].aabb.contains(aabb)) return false;

    // Remove the current leaf.
    removeLeaf(node);

    // Fatten the new AABB.
    for (unsigned int i = 0; i < dimension; i++) {
      aabb.m_min[i] -= m_skinThickness * size[i];
      aabb.m_max[i] += m_skinThickness * size[i];
    }

    // Assign the new AABB.
    m_nodes[node].aabb = aabb;

    // Update the surface area and centroid.
    m_nodes[node].aabb.m_area = m_nodes[node].aabb.compute_area();
    //    m_nodes[node].aabb.m_centre = m_nodes[node].aabb.computeCentre();

    // Insert a new leaf node.
    insertLeaf(node);

#ifndef NDEBUG
    validate();
#endif

    return true;
  }

  [[nodiscard]] auto query(const key_type& id) const -> std::vector<key_type>
  {
    // Make sure that this is a valid particle.
    if (m_indexMap.count(id) == 0) {
      throw std::invalid_argument("[ERROR]: Invalid particle index!");
    }

    // Test overlap of particle AABB against all other particles.
    return query(id, m_nodes[m_indexMap.find(id)->second].aabb);
  }

  [[nodiscard]] auto query(const key_type& id, const aabb_type& aabb) const
      -> std::vector<key_type>
  {
    std::vector<index_type> stack;
    stack.reserve(256);
    stack.push_back(m_root);

    std::vector<key_type> particles;

    while (stack.size() > 0) {
      const auto node = stack.back();
      stack.pop_back();

      // Copy the AABB.
      aabb_type nodeAABB = m_nodes[node].aabb;

      if (node == NULL_NODE) continue;

      // Test for overlap between the AABBs.
      if (aabb.overlaps(nodeAABB, m_touchIsOverlap)) {
        // Check that we're at a leaf node.
        if (m_nodes[node].is_leaf()) {
          // Can't interact with itself.
          if (m_nodes[node].id != id) {
            particles.push_back(m_nodes[node].id);
          }
        } else {
          stack.push_back(m_nodes[node].left);
          stack.push_back(m_nodes[node].right);
        }
      }
    }

    return particles;
  }

  [[nodiscard]] auto query(const aabb_type& aabb) const -> std::vector<key_type>
  {
    // Make sure the tree isn't empty.
    if (m_indexMap.size() == 0) {
      return std::vector<unsigned int>();
    }

    // Test overlap of AABB against all particles.
    return query(std::numeric_limits<unsigned int>::max(), aabb);
  }

  [[nodiscard]] auto compute_maximum_balance() const -> size_type
  {
    size_type maxBalance{0};
    for (auto i = 0; i < m_nodeCapacity; ++i) {
      // if (node.height <= 1) {
      //   continue;
      // }
      const auto& node = m_nodes.at(i);
      if (node.height > 2) {
        assert(!node.is_leaf());
        assert(node.left != NULL_NODE);
        assert(node.right != NULL_NODE);

        const auto balance = std::abs(m_nodes.at(node.left).height -
                                      m_nodes.at(node.right).height);
        maxBalance = std::max(maxBalance, balance);
      }
    }

    return maxBalance;
  }

  [[nodiscard]] auto computeSurfaceAreaRatio() const -> double
  {
    if (m_root == NULL_NODE) return 0.0;

    double rootArea = m_nodes[m_root].aabb.compute_area();
    double totalArea = 0.0;

    for (unsigned int i = 0; i < m_nodeCapacity; i++) {
      if (m_nodes[i].height < 0) continue;

      totalArea += m_nodes[i].aabb.compute_area();
    }

    return totalArea / rootArea;
  }

  /// Rebuild an optimal tree.
  void rebuild()
  {
    std::vector<index_type> nodeIndices(m_nodeCount);
    unsigned int count = 0;

    for (unsigned int i = 0; i < m_nodeCapacity; i++) {
      // Free node.
      if (m_nodes[i].height < 0) continue;

      if (m_nodes[i].is_leaf()) {
        m_nodes[i].parent = NULL_NODE;
        nodeIndices[count] = i;
        count++;
      } else
        freeNode(i);
    }

    while (count > 1) {
      double minCost = std::numeric_limits<double>::max();
      int iMin = -1, jMin = -1;

      for (unsigned int i = 0; i < count; i++) {
        aabb_type aabbi = m_nodes[nodeIndices[i]].aabb;

        for (unsigned int j = i + 1; j < count; j++) {
          aabb_type aabbj = m_nodes[nodeIndices[j]].aabb;
          aabb_type aabb;
          aabb.merge(aabbi, aabbj);
          double cost = aabb.area();

          if (cost < minCost) {
            iMin = i;
            jMin = j;
            minCost = cost;
          }
        }
      }

      unsigned int index1 = nodeIndices[iMin];
      unsigned int index2 = nodeIndices[jMin];

      unsigned int parent = allocateNode();
      m_nodes[parent].left = index1;
      m_nodes[parent].right = index2;
      m_nodes[parent].height =
          1 + std::max(m_nodes[index1].height, m_nodes[index2].height);
      m_nodes[parent].aabb.merge(m_nodes[index1].aabb, m_nodes[index2].aabb);
      m_nodes[parent].parent = NULL_NODE;

      m_nodes[index1].parent = parent;
      m_nodes[index2].parent = parent;

      nodeIndices[jMin] = nodeIndices[count - 1];
      nodeIndices[iMin] = parent;
      count--;
    }

    m_root = nodeIndices[0];

    validate();
  }

  void validate() const
  {
#ifndef NDEBUG
    validateStructure(m_root);
    validateMetrics(m_root);

    unsigned int freeCount = 0;
    unsigned int freeIndex = m_nextFreeIndex;

    while (freeIndex != NULL_NODE) {
      assert(freeIndex < m_nodeCapacity);
      freeIndex = m_nodes[freeIndex].next;
      freeCount++;
    }

    assert(height() == computeHeight());
    assert((m_nodeCount + freeCount) == m_nodeCapacity);
#endif
  }

  [[nodiscard]] auto get_aabb(const key_type& id) const -> const aabb_type&
  {
    return m_nodes[m_indexMap[id]].aabb;
  }

  [[nodiscard]] auto height() const -> unsigned int
  {
    if (m_root == NULL_NODE) return 0;
    return m_nodes[m_root].height;
  }

  [[nodiscard]] auto getNodeCount() const noexcept -> unsigned int
  {
    return m_nodeCount;
  }

  [[nodiscard]] auto nParticles() const noexcept -> size_type
  {
    return m_indexMap.size();
  }

 private:
  std::vector<node_type> m_nodes;
  std::unordered_map<key_type, index_type> m_indexMap;

  index_type m_root{NULL_NODE};   ///< Root node index
  index_type m_nextFreeIndex{0};  ///< Index of next free node

  size_type m_nodeCount{0};  ///< Number of m_nodes in the tree.
  size_type m_nodeCapacity;  ///< Current node capacity.

  double m_skinThickness{0.05};

  /// Does touching count as overlapping in tree queries?
  bool m_touchIsOverlap;

  [[nodiscard]] auto allocateNode() -> index_type
  {
    // Exand the node pool as needed.
    if (m_nextFreeIndex == NULL_NODE) {
      assert(m_nodeCount == m_nodeCapacity);

      // The free list is empty. Rebuild a bigger pool.
      m_nodeCapacity *= 2;
      m_nodes.resize(m_nodeCapacity);

      // Build a linked list for the list of free m_nodes.
      for (unsigned int i = m_nodeCount; i < m_nodeCapacity - 1; i++) {
        m_nodes[i].next = i + 1;
        m_nodes[i].height = -1;
      }
      m_nodes[m_nodeCapacity - 1].next = NULL_NODE;
      m_nodes[m_nodeCapacity - 1].height = -1;

      // Assign the index of the first free node.
      m_nextFreeIndex = m_nodeCount;
    }

    // Peel a node off the free list.
    const auto node = m_nextFreeIndex;
    m_nextFreeIndex = m_nodes[node].next;
    m_nodes[node].parent = NULL_NODE;
    m_nodes[node].left = NULL_NODE;
    m_nodes[node].right = NULL_NODE;
    m_nodes[node].height = 0;
    //    m_nodes[node].aabb.set_dimension(dimension);
    m_nodeCount++;

    return node;
  }

  //! Free an existing node.
  /*! \param node
          The index of the node to be freed.
   */
  void freeNode(index_type node)
  {
    assert(node < m_nodeCapacity);
    assert(0 < m_nodeCount);

    m_nodes[node].next = m_nextFreeIndex;
    m_nodes[node].height = -1;
    m_nextFreeIndex = node;
    m_nodeCount--;
  }

  void insertLeaf(index_type leaf)
  {
    if (m_root == NULL_NODE) {
      m_root = leaf;
      m_nodes[m_root].parent = NULL_NODE;
      return;
    }

    // Find the best sibling for the node.

    const aabb_type leafAABB = m_nodes[leaf].aabb;
    unsigned int index = m_root;

    while (!m_nodes[index].is_leaf()) {
      // Extract the children of the node.
      unsigned int left = m_nodes[index].left;
      unsigned int right = m_nodes[index].right;

      double surfaceArea = m_nodes[index].aabb.area();

      aabb_type combinedAABB;
      combinedAABB.merge(m_nodes[index].aabb, leafAABB);
      double combinedSurfaceArea = combinedAABB.area();

      // Cost of creating a new parent for this node and the new leaf.
      double cost = 2.0 * combinedSurfaceArea;

      // Minimum cost of pushing the leaf further down the tree.
      double inheritanceCost = 2.0 * (combinedSurfaceArea - surfaceArea);

      // Cost of descending to the left.
      double costLeft;
      if (m_nodes[left].is_leaf()) {
        aabb_type aabb;
        aabb.merge(leafAABB, m_nodes[left].aabb);
        costLeft = aabb.area() + inheritanceCost;
      } else {
        aabb_type aabb;
        aabb.merge(leafAABB, m_nodes[left].aabb);
        double oldArea = m_nodes[left].aabb.area();
        double newArea = aabb.area();
        costLeft = (newArea - oldArea) + inheritanceCost;
      }

      // Cost of descending to the right.
      double costRight;
      if (m_nodes[right].is_leaf()) {
        aabb_type aabb;
        aabb.merge(leafAABB, m_nodes[right].aabb);
        costRight = aabb.area() + inheritanceCost;
      } else {
        aabb_type aabb;
        aabb.merge(leafAABB, m_nodes[right].aabb);
        double oldArea = m_nodes[right].aabb.area();
        double newArea = aabb.area();
        costRight = (newArea - oldArea) + inheritanceCost;
      }

      // Descend according to the minimum cost.
      if ((cost < costLeft) && (cost < costRight)) break;

      // Descend.
      if (costLeft < costRight)
        index = left;
      else
        index = right;
    }

    unsigned int sibling = index;

    // Create a new parent.
    unsigned int oldParent = m_nodes[sibling].parent;
    unsigned int newParent = allocateNode();
    m_nodes[newParent].parent = oldParent;
    m_nodes[newParent].aabb.merge(leafAABB, m_nodes[sibling].aabb);
    m_nodes[newParent].height = m_nodes[sibling].height + 1;

    // The sibling was not the root.
    if (oldParent != NULL_NODE) {
      if (m_nodes[oldParent].left == sibling)
        m_nodes[oldParent].left = newParent;
      else
        m_nodes[oldParent].right = newParent;

      m_nodes[newParent].left = sibling;
      m_nodes[newParent].right = leaf;
      m_nodes[sibling].parent = newParent;
      m_nodes[leaf].parent = newParent;
    }
    // The sibling was the root.
    else {
      m_nodes[newParent].left = sibling;
      m_nodes[newParent].right = leaf;
      m_nodes[sibling].parent = newParent;
      m_nodes[leaf].parent = newParent;
      m_root = newParent;
    }

    // Walk back up the tree fixing heights and AABBs.
    index = m_nodes[leaf].parent;
    while (index != NULL_NODE) {
      index = balance(index);

      unsigned int left = m_nodes[index].left;
      unsigned int right = m_nodes[index].right;

      assert(left != NULL_NODE);
      assert(right != NULL_NODE);

      m_nodes[index].height =
          1 + std::max(m_nodes[left].height, m_nodes[right].height);
      m_nodes[index].aabb.merge(m_nodes[left].aabb, m_nodes[right].aabb);

      index = m_nodes[index].parent;
    }
  }

  void removeLeaf(index_type leaf)
  {
    if (leaf == m_root) {
      m_root = NULL_NODE;
      return;
    }

    unsigned int parent = m_nodes[leaf].parent;
    unsigned int grandParent = m_nodes[parent].parent;
    unsigned int sibling;

    if (m_nodes[parent].left == leaf)
      sibling = m_nodes[parent].right;
    else
      sibling = m_nodes[parent].left;

    // Destroy the parent and connect the sibling to the grandparent.
    if (grandParent != NULL_NODE) {
      if (m_nodes[grandParent].left == parent)
        m_nodes[grandParent].left = sibling;
      else
        m_nodes[grandParent].right = sibling;

      m_nodes[sibling].parent = grandParent;
      freeNode(parent);

      // Adjust ancestor bounds.
      unsigned int index = grandParent;
      while (index != NULL_NODE) {
        index = balance(index);

        unsigned int left = m_nodes[index].left;
        unsigned int right = m_nodes[index].right;

        m_nodes[index].aabb.merge(m_nodes[left].aabb, m_nodes[right].aabb);
        m_nodes[index].height =
            1 + std::max(m_nodes[left].height, m_nodes[right].height);

        index = m_nodes[index].parent;
      }
    } else {
      m_root = sibling;
      m_nodes[sibling].parent = NULL_NODE;
      freeNode(parent);
    }
  }

  unsigned int balance(index_type node)
  {
    assert(node != NULL_NODE);

    if (m_nodes[node].is_leaf() || (m_nodes[node].height < 2)) return node;

    unsigned int left = m_nodes[node].left;
    unsigned int right = m_nodes[node].right;

    assert(left < m_nodeCapacity);
    assert(right < m_nodeCapacity);

    int currentBalance = m_nodes[right].height - m_nodes[left].height;

    // Rotate right branch up.
    if (currentBalance > 1) {
      unsigned int rightLeft = m_nodes[right].left;
      unsigned int rightRight = m_nodes[right].right;

      assert(rightLeft < m_nodeCapacity);
      assert(rightRight < m_nodeCapacity);

      // Swap node and its right-hand child.
      m_nodes[right].left = node;
      m_nodes[right].parent = m_nodes[node].parent;
      m_nodes[node].parent = right;

      // The node's old parent should now point to its right-hand child.
      if (m_nodes[right].parent != NULL_NODE) {
        if (m_nodes[m_nodes[right].parent].left == node)
          m_nodes[m_nodes[right].parent].left = right;
        else {
          assert(m_nodes[m_nodes[right].parent].right == node);
          m_nodes[m_nodes[right].parent].right = right;
        }
      } else
        m_root = right;

      // Rotate.
      if (m_nodes[rightLeft].height > m_nodes[rightRight].height) {
        m_nodes[right].right = rightLeft;
        m_nodes[node].right = rightRight;
        m_nodes[rightRight].parent = node;
        m_nodes[node].aabb.merge(m_nodes[left].aabb, m_nodes[rightRight].aabb);
        m_nodes[right].aabb.merge(m_nodes[node].aabb, m_nodes[rightLeft].aabb);

        m_nodes[node].height =
            1 + std::max(m_nodes[left].height, m_nodes[rightRight].height);
        m_nodes[right].height =
            1 + std::max(m_nodes[node].height, m_nodes[rightLeft].height);
      } else {
        m_nodes[right].right = rightRight;
        m_nodes[node].right = rightLeft;
        m_nodes[rightLeft].parent = node;
        m_nodes[node].aabb.merge(m_nodes[left].aabb, m_nodes[rightLeft].aabb);
        m_nodes[right].aabb.merge(m_nodes[node].aabb, m_nodes[rightRight].aabb);

        m_nodes[node].height =
            1 + std::max(m_nodes[left].height, m_nodes[rightLeft].height);
        m_nodes[right].height =
            1 + std::max(m_nodes[node].height, m_nodes[rightRight].height);
      }

      return right;
    }

    // Rotate left branch up.
    if (currentBalance < -1) {
      unsigned int leftLeft = m_nodes[left].left;
      unsigned int leftRight = m_nodes[left].right;

      assert(leftLeft < m_nodeCapacity);
      assert(leftRight < m_nodeCapacity);

      // Swap node and its left-hand child.
      m_nodes[left].left = node;
      m_nodes[left].parent = m_nodes[node].parent;
      m_nodes[node].parent = left;

      // The node's old parent should now point to its left-hand child.
      if (m_nodes[left].parent != NULL_NODE) {
        if (m_nodes[m_nodes[left].parent].left == node)
          m_nodes[m_nodes[left].parent].left = left;
        else {
          assert(m_nodes[m_nodes[left].parent].right == node);
          m_nodes[m_nodes[left].parent].right = left;
        }
      } else
        m_root = left;

      // Rotate.
      if (m_nodes[leftLeft].height > m_nodes[leftRight].height) {
        m_nodes[left].right = leftLeft;
        m_nodes[node].left = leftRight;
        m_nodes[leftRight].parent = node;
        m_nodes[node].aabb.merge(m_nodes[right].aabb, m_nodes[leftRight].aabb);
        m_nodes[left].aabb.merge(m_nodes[node].aabb, m_nodes[leftLeft].aabb);

        m_nodes[node].height =
            1 + std::max(m_nodes[right].height, m_nodes[leftRight].height);
        m_nodes[left].height =
            1 + std::max(m_nodes[node].height, m_nodes[leftLeft].height);
      } else {
        m_nodes[left].right = leftRight;
        m_nodes[node].left = leftLeft;
        m_nodes[leftLeft].parent = node;
        m_nodes[node].aabb.merge(m_nodes[right].aabb, m_nodes[leftLeft].aabb);
        m_nodes[left].aabb.merge(m_nodes[node].aabb, m_nodes[leftRight].aabb);

        m_nodes[node].height =
            1 + std::max(m_nodes[right].height, m_nodes[leftLeft].height);
        m_nodes[left].height =
            1 + std::max(m_nodes[node].height, m_nodes[leftRight].height);
      }

      return left;
    }

    return node;
  }

  [[nodiscard]] auto computeHeight() const -> size_type
  {
    return computeHeight(m_root);
  }

  [[nodiscard]] auto computeHeight(index_type node) const -> size_type
  {
    assert(node < m_nodeCapacity);

    if (m_nodes[node].is_leaf()) return 0;

    unsigned int height1 = computeHeight(m_nodes[node].left);
    unsigned int height2 = computeHeight(m_nodes[node].right);

    return 1 + std::max(height1, height2);
  }

  //! Assert that the sub-tree has a valid structure.
  /*! \param node
          The index of the root node.
   */
  void validateStructure(index_type node) const
  {
    if (node == NULL_NODE) return;

    if (node == m_root) assert(m_nodes[node].parent == NULL_NODE);

    unsigned int left = m_nodes[node].left;
    unsigned int right = m_nodes[node].right;

    if (m_nodes[node].is_leaf()) {
      assert(left == NULL_NODE);
      assert(right == NULL_NODE);
      assert(m_nodes[node].height == 0);
      return;
    }

    assert(left < m_nodeCapacity);
    assert(right < m_nodeCapacity);

    assert(m_nodes[left].parent == node);
    assert(m_nodes[right].parent == node);

    validateStructure(left);
    validateStructure(right);
  }

  //! Assert that the sub-tree has valid metrics.
  /*! \param node
          The index of the root node.
   */
  void validateMetrics(index_type node) const
  {
    if (node == NULL_NODE) return;

    unsigned int left = m_nodes[node].left;
    unsigned int right = m_nodes[node].right;

    if (m_nodes[node].is_leaf()) {
      assert(left == NULL_NODE);
      assert(right == NULL_NODE);
      assert(m_nodes[node].height == 0);
      return;
    }

    assert(left < m_nodeCapacity);
    assert(right < m_nodeCapacity);

    int height1 = m_nodes[left].height;
    int height2 = m_nodes[right].height;
    int height = 1 + std::max(height1, height2);
    (void)height;  // Unused variable in Release build
    assert(m_nodes[node].height == height);

    aabb_type aabb;
    aabb.merge(m_nodes[left].aabb, m_nodes[right].aabb);

    for (unsigned int i = 0; i < 2; i++) {
      assert(aabb.m_min[i] == m_nodes[node].aabb.m_min[i]);
      assert(aabb.m_max[i] == m_nodes[node].aabb.m_max[i]);
    }

    validateMetrics(left);
    validateMetrics(right);
  }
};

}  // namespace abby2
