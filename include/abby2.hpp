#pragma once

#include <algorithm>      // min, max
#include <cassert>        // assert
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

  explicit tree(size_type dimension_ = 3,
                double skinThickness_ = 0.05,
                size_type nParticles = 16,
                bool touchIsOverlap = true)
      : dimension{dimension_},
        skinThickness{skinThickness_},
        touchIsOverlap{touchIsOverlap}
  {
    // Validate the dimensionality.
    if ((dimension < 2)) {
      throw std::invalid_argument("[ERROR]: Invalid dimensionality!");
    }

    // Initialise the tree.
    root = NULL_NODE;
    nodeCount = 0;
    nodeCapacity = nParticles;
    nodes.resize(nodeCapacity);

    // Build a linked list for the list of free nodes.
    for (auto i = 0; i < nodeCapacity - 1; i++) {
      nodes[i].next = i + 1;
      nodes[i].height = -1;
    }
    nodes[nodeCapacity - 1].next = NULL_NODE;
    nodes[nodeCapacity - 1].height = -1;

    // Assign the index of the first free node.
    freeList = 0;
  }

  void insertParticle(const key_type& particle,
                      vector_type& lowerBound,
                      vector_type& upperBound)
  {
    // Make sure the particle doesn't already exist.
    if (particleMap.count(particle) != 0) {
      throw std::invalid_argument("[ERROR]: Particle already exists in tree!");
    }

    // Allocate a new node for the particle.
    unsigned int node = allocateNode();

    // AABB size in each dimension.
    std::vector<double> size(dimension);

    // Compute the AABB limits.
    for (unsigned int i = 0; i < dimension; i++) {
      // Validate the bound.
      if (lowerBound[i] > upperBound[i]) {
        throw std::invalid_argument(
            "[ERROR]: AABB lower bound is greater than the upper bound!");
      }

      nodes[node].aabb.m_min[i] = lowerBound[i];
      nodes[node].aabb.m_max[i] = upperBound[i];
      size[i] = upperBound[i] - lowerBound[i];
    }

    // Fatten the AABB.
    for (unsigned int i = 0; i < dimension; i++) {
      nodes[node].aabb.m_min[i] -= skinThickness * size[i];
      nodes[node].aabb.m_max[i] += skinThickness * size[i];
    }
    nodes[node].aabb.m_area = nodes[node].aabb.compute_area();
    //    nodes[node].aabb.m_centre = nodes[node].aabb.computeCentre();

    // Zero the height.
    nodes[node].height = 0;

    // Insert a new leaf into the tree.
    insertLeaf(node);

    // Add the new particle to the map.
    particleMap.insert(
        std::unordered_map<unsigned int, unsigned int>::value_type(particle,
                                                                   node));

    // Store the particle index.
    nodes[node].id = particle;

#ifndef NDEBUG
    validate();
#endif
  }

  void removeParticle(const key_type& particle)
  {
    // Map iterator.
    std::unordered_map<unsigned int, unsigned int>::iterator it;

    // Find the particle.
    it = particleMap.find(particle);

    // The particle doesn't exist.
    if (it == particleMap.end()) {
      throw std::invalid_argument("[ERROR]: Invalid particle index!");
    }

    // Extract the node index.
    unsigned int node = it->second;

    // Erase the particle from the map.
    particleMap.erase(it);

    assert(node < nodeCapacity);
    assert(nodes[node].is_leaf());

    removeLeaf(node);
    freeNode(node);

#ifndef NDEBUG
    validate();
#endif
  }

  void removeAll()
  {
    // Iterator pointing to the start of the particle map.
    std::unordered_map<unsigned int, unsigned int>::iterator it =
        particleMap.begin();

    // Iterate over the map.
    while (it != particleMap.end()) {
      // Extract the node index.
      unsigned int node = it->second;

      assert(node < nodeCapacity);
      assert(nodes[node].is_leaf());

      removeLeaf(node);
      freeNode(node);

      it++;
    }

    // Clear the particle map.
    particleMap.clear();

#ifndef NDEBUG
    validate();
#endif
  }

  void print(std::ostream& stream) const
  {
    stream << "abby2:\n";
    print(stream, "", root, false);
  }

  void print(std::ostream& stream,
             const std::string& prefix,
             index_type index,
             bool isLeft) const
  {
    if (index != NULL_NODE) {
      const auto& node = nodes.at(index);

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

  auto updateParticle(const key_type& particle,
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
    it = particleMap.find(particle);

    // The particle doesn't exist.
    if (it == particleMap.end()) {
      throw std::invalid_argument("[ERROR]: Invalid particle index!");
    }

    // Extract the node index.
    unsigned int node = it->second;

    assert(node < nodeCapacity);
    assert(nodes[node].is_leaf());

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
    if (!alwaysReinsert && nodes[node].aabb.contains(aabb)) return false;

    // Remove the current leaf.
    removeLeaf(node);

    // Fatten the new AABB.
    for (unsigned int i = 0; i < dimension; i++) {
      aabb.m_min[i] -= skinThickness * size[i];
      aabb.m_max[i] += skinThickness * size[i];
    }

    // Assign the new AABB.
    nodes[node].aabb = aabb;

    // Update the surface area and centroid.
    nodes[node].aabb.m_area = nodes[node].aabb.compute_area();
    //    nodes[node].aabb.m_centre = nodes[node].aabb.computeCentre();

    // Insert a new leaf node.
    insertLeaf(node);

#ifndef NDEBUG
    validate();
#endif

    return true;
  }

  [[nodiscard]] auto query(const key_type& particle) const
      -> std::vector<key_type>
  {
    // Make sure that this is a valid particle.
    if (particleMap.count(particle) == 0) {
      throw std::invalid_argument("[ERROR]: Invalid particle index!");
    }

    // Test overlap of particle AABB against all other particles.
    return query(particle, nodes[particleMap.find(particle)->second].aabb);
  }

  [[nodiscard]] auto query(const key_type& particle,
                           const aabb_type& aabb) const -> std::vector<key_type>
  {
    std::vector<index_type> stack;
    stack.reserve(256);
    stack.push_back(root);

    std::vector<key_type> particles;

    while (stack.size() > 0) {
      const auto node = stack.back();
      stack.pop_back();

      // Copy the AABB.
      aabb_type nodeAABB = nodes[node].aabb;

      if (node == NULL_NODE) continue;

      // Test for overlap between the AABBs.
      if (aabb.overlaps(nodeAABB, touchIsOverlap)) {
        // Check that we're at a leaf node.
        if (nodes[node].is_leaf()) {
          // Can't interact with itself.
          if (nodes[node].id != particle) {
            particles.push_back(nodes[node].id);
          }
        } else {
          stack.push_back(nodes[node].left);
          stack.push_back(nodes[node].right);
        }
      }
    }

    return particles;
  }

  [[nodiscard]] auto query(const aabb_type& aabb) const -> std::vector<key_type>
  {
    // Make sure the tree isn't empty.
    if (particleMap.size() == 0) {
      return std::vector<unsigned int>();
    }

    // Test overlap of AABB against all particles.
    return query(std::numeric_limits<unsigned int>::max(), aabb);
  }

  [[nodiscard]] auto computeMaximumBalance() const -> unsigned int
  {
    unsigned int maxBalance = 0;
    for (unsigned int i = 0; i < nodeCapacity; i++) {
      if (nodes[i].height <= 1) continue;

      assert(nodes[i].is_leaf() == false);

      unsigned int balance =
          std::abs(nodes[nodes[i].left].height - nodes[nodes[i].right].height);
      maxBalance = std::max(maxBalance, balance);
    }

    return maxBalance;
  }

  [[nodiscard]] auto computeSurfaceAreaRatio() const -> double
  {
    if (root == NULL_NODE) return 0.0;

    double rootArea = nodes[root].aabb.compute_area();
    double totalArea = 0.0;

    for (unsigned int i = 0; i < nodeCapacity; i++) {
      if (nodes[i].height < 0) continue;

      totalArea += nodes[i].aabb.compute_area();
    }

    return totalArea / rootArea;
  }

  /// Rebuild an optimal tree.
  void rebuild()
  {
    std::vector<index_type> nodeIndices(nodeCount);
    unsigned int count = 0;

    for (unsigned int i = 0; i < nodeCapacity; i++) {
      // Free node.
      if (nodes[i].height < 0) continue;

      if (nodes[i].is_leaf()) {
        nodes[i].parent = NULL_NODE;
        nodeIndices[count] = i;
        count++;
      } else
        freeNode(i);
    }

    while (count > 1) {
      double minCost = std::numeric_limits<double>::max();
      int iMin = -1, jMin = -1;

      for (unsigned int i = 0; i < count; i++) {
        aabb_type aabbi = nodes[nodeIndices[i]].aabb;

        for (unsigned int j = i + 1; j < count; j++) {
          aabb_type aabbj = nodes[nodeIndices[j]].aabb;
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
      nodes[parent].left = index1;
      nodes[parent].right = index2;
      nodes[parent].height =
          1 + std::max(nodes[index1].height, nodes[index2].height);
      nodes[parent].aabb.merge(nodes[index1].aabb, nodes[index2].aabb);
      nodes[parent].parent = NULL_NODE;

      nodes[index1].parent = parent;
      nodes[index2].parent = parent;

      nodeIndices[jMin] = nodeIndices[count - 1];
      nodeIndices[iMin] = parent;
      count--;
    }

    root = nodeIndices[0];

    validate();
  }

  void validate() const
  {
#ifndef NDEBUG
    validateStructure(root);
    validateMetrics(root);

    unsigned int freeCount = 0;
    unsigned int freeIndex = freeList;

    while (freeIndex != NULL_NODE) {
      assert(freeIndex < nodeCapacity);
      freeIndex = nodes[freeIndex].next;
      freeCount++;
    }

    assert(getHeight() == computeHeight());
    assert((nodeCount + freeCount) == nodeCapacity);
#endif
  }

  [[nodiscard]] auto getAABB(const key_type& particle) const -> const aabb_type&
  {
    return nodes[particleMap[particle]].aabb;
  }

  [[nodiscard]] auto getHeight() const -> unsigned int
  {
    if (root == NULL_NODE) return 0;
    return nodes[root].height;
  }

  [[nodiscard]] auto getNodeCount() const noexcept -> unsigned int
  {
    return nodeCount;
  }

  [[nodiscard]] auto nParticles() const noexcept -> size_type
  {
    return particleMap.size();
  }

 private:
  /// The index of the root node.
  unsigned int root;

  /// The dynamic tree.
  std::vector<node_type> nodes;

  /// The current number of nodes in the tree.
  unsigned int nodeCount;

  /// The current node capacity.
  unsigned int nodeCapacity;

  /// The position of node at the top of the free list.
  unsigned int freeList;

  /// The dimensionality of the system.
  size_type dimension;

  /// The skin thickness of the fattened AABBs, as a fraction of the AABB base
  /// length.
  double skinThickness;

  /// A map between particle and node indices.
  std::unordered_map<key_type, index_type> particleMap;

  /// Does touching count as overlapping in tree queries?
  bool touchIsOverlap;

  [[nodiscard]] auto allocateNode() -> index_type
  {
    // Exand the node pool as needed.
    if (freeList == NULL_NODE) {
      assert(nodeCount == nodeCapacity);

      // The free list is empty. Rebuild a bigger pool.
      nodeCapacity *= 2;
      nodes.resize(nodeCapacity);

      // Build a linked list for the list of free nodes.
      for (unsigned int i = nodeCount; i < nodeCapacity - 1; i++) {
        nodes[i].next = i + 1;
        nodes[i].height = -1;
      }
      nodes[nodeCapacity - 1].next = NULL_NODE;
      nodes[nodeCapacity - 1].height = -1;

      // Assign the index of the first free node.
      freeList = nodeCount;
    }

    // Peel a node off the free list.
    const auto node = freeList;
    freeList = nodes[node].next;
    nodes[node].parent = NULL_NODE;
    nodes[node].left = NULL_NODE;
    nodes[node].right = NULL_NODE;
    nodes[node].height = 0;
    //    nodes[node].aabb.set_dimension(dimension);
    nodeCount++;

    return node;
  }

  //! Free an existing node.
  /*! \param node
          The index of the node to be freed.
   */
  void freeNode(index_type node)
  {
    assert(node < nodeCapacity);
    assert(0 < nodeCount);

    nodes[node].next = freeList;
    nodes[node].height = -1;
    freeList = node;
    nodeCount--;
  }

  void insertLeaf(index_type leaf)
  {
    if (root == NULL_NODE) {
      root = leaf;
      nodes[root].parent = NULL_NODE;
      return;
    }

    // Find the best sibling for the node.

    const aabb_type leafAABB = nodes[leaf].aabb;
    unsigned int index = root;

    while (!nodes[index].is_leaf()) {
      // Extract the children of the node.
      unsigned int left = nodes[index].left;
      unsigned int right = nodes[index].right;

      double surfaceArea = nodes[index].aabb.area();

      aabb_type combinedAABB;
      combinedAABB.merge(nodes[index].aabb, leafAABB);
      double combinedSurfaceArea = combinedAABB.area();

      // Cost of creating a new parent for this node and the new leaf.
      double cost = 2.0 * combinedSurfaceArea;

      // Minimum cost of pushing the leaf further down the tree.
      double inheritanceCost = 2.0 * (combinedSurfaceArea - surfaceArea);

      // Cost of descending to the left.
      double costLeft;
      if (nodes[left].is_leaf()) {
        aabb_type aabb;
        aabb.merge(leafAABB, nodes[left].aabb);
        costLeft = aabb.area() + inheritanceCost;
      } else {
        aabb_type aabb;
        aabb.merge(leafAABB, nodes[left].aabb);
        double oldArea = nodes[left].aabb.area();
        double newArea = aabb.area();
        costLeft = (newArea - oldArea) + inheritanceCost;
      }

      // Cost of descending to the right.
      double costRight;
      if (nodes[right].is_leaf()) {
        aabb_type aabb;
        aabb.merge(leafAABB, nodes[right].aabb);
        costRight = aabb.area() + inheritanceCost;
      } else {
        aabb_type aabb;
        aabb.merge(leafAABB, nodes[right].aabb);
        double oldArea = nodes[right].aabb.area();
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
    unsigned int oldParent = nodes[sibling].parent;
    unsigned int newParent = allocateNode();
    nodes[newParent].parent = oldParent;
    nodes[newParent].aabb.merge(leafAABB, nodes[sibling].aabb);
    nodes[newParent].height = nodes[sibling].height + 1;

    // The sibling was not the root.
    if (oldParent != NULL_NODE) {
      if (nodes[oldParent].left == sibling)
        nodes[oldParent].left = newParent;
      else
        nodes[oldParent].right = newParent;

      nodes[newParent].left = sibling;
      nodes[newParent].right = leaf;
      nodes[sibling].parent = newParent;
      nodes[leaf].parent = newParent;
    }
    // The sibling was the root.
    else {
      nodes[newParent].left = sibling;
      nodes[newParent].right = leaf;
      nodes[sibling].parent = newParent;
      nodes[leaf].parent = newParent;
      root = newParent;
    }

    // Walk back up the tree fixing heights and AABBs.
    index = nodes[leaf].parent;
    while (index != NULL_NODE) {
      index = balance(index);

      unsigned int left = nodes[index].left;
      unsigned int right = nodes[index].right;

      assert(left != NULL_NODE);
      assert(right != NULL_NODE);

      nodes[index].height =
          1 + std::max(nodes[left].height, nodes[right].height);
      nodes[index].aabb.merge(nodes[left].aabb, nodes[right].aabb);

      index = nodes[index].parent;
    }
  }

  void removeLeaf(index_type leaf)
  {
    if (leaf == root) {
      root = NULL_NODE;
      return;
    }

    unsigned int parent = nodes[leaf].parent;
    unsigned int grandParent = nodes[parent].parent;
    unsigned int sibling;

    if (nodes[parent].left == leaf)
      sibling = nodes[parent].right;
    else
      sibling = nodes[parent].left;

    // Destroy the parent and connect the sibling to the grandparent.
    if (grandParent != NULL_NODE) {
      if (nodes[grandParent].left == parent)
        nodes[grandParent].left = sibling;
      else
        nodes[grandParent].right = sibling;

      nodes[sibling].parent = grandParent;
      freeNode(parent);

      // Adjust ancestor bounds.
      unsigned int index = grandParent;
      while (index != NULL_NODE) {
        index = balance(index);

        unsigned int left = nodes[index].left;
        unsigned int right = nodes[index].right;

        nodes[index].aabb.merge(nodes[left].aabb, nodes[right].aabb);
        nodes[index].height =
            1 + std::max(nodes[left].height, nodes[right].height);

        index = nodes[index].parent;
      }
    } else {
      root = sibling;
      nodes[sibling].parent = NULL_NODE;
      freeNode(parent);
    }
  }

  unsigned int balance(index_type node)
  {
    assert(node != NULL_NODE);

    if (nodes[node].is_leaf() || (nodes[node].height < 2)) return node;

    unsigned int left = nodes[node].left;
    unsigned int right = nodes[node].right;

    assert(left < nodeCapacity);
    assert(right < nodeCapacity);

    int currentBalance = nodes[right].height - nodes[left].height;

    // Rotate right branch up.
    if (currentBalance > 1) {
      unsigned int rightLeft = nodes[right].left;
      unsigned int rightRight = nodes[right].right;

      assert(rightLeft < nodeCapacity);
      assert(rightRight < nodeCapacity);

      // Swap node and its right-hand child.
      nodes[right].left = node;
      nodes[right].parent = nodes[node].parent;
      nodes[node].parent = right;

      // The node's old parent should now point to its right-hand child.
      if (nodes[right].parent != NULL_NODE) {
        if (nodes[nodes[right].parent].left == node)
          nodes[nodes[right].parent].left = right;
        else {
          assert(nodes[nodes[right].parent].right == node);
          nodes[nodes[right].parent].right = right;
        }
      } else
        root = right;

      // Rotate.
      if (nodes[rightLeft].height > nodes[rightRight].height) {
        nodes[right].right = rightLeft;
        nodes[node].right = rightRight;
        nodes[rightRight].parent = node;
        nodes[node].aabb.merge(nodes[left].aabb, nodes[rightRight].aabb);
        nodes[right].aabb.merge(nodes[node].aabb, nodes[rightLeft].aabb);

        nodes[node].height =
            1 + std::max(nodes[left].height, nodes[rightRight].height);
        nodes[right].height =
            1 + std::max(nodes[node].height, nodes[rightLeft].height);
      } else {
        nodes[right].right = rightRight;
        nodes[node].right = rightLeft;
        nodes[rightLeft].parent = node;
        nodes[node].aabb.merge(nodes[left].aabb, nodes[rightLeft].aabb);
        nodes[right].aabb.merge(nodes[node].aabb, nodes[rightRight].aabb);

        nodes[node].height =
            1 + std::max(nodes[left].height, nodes[rightLeft].height);
        nodes[right].height =
            1 + std::max(nodes[node].height, nodes[rightRight].height);
      }

      return right;
    }

    // Rotate left branch up.
    if (currentBalance < -1) {
      unsigned int leftLeft = nodes[left].left;
      unsigned int leftRight = nodes[left].right;

      assert(leftLeft < nodeCapacity);
      assert(leftRight < nodeCapacity);

      // Swap node and its left-hand child.
      nodes[left].left = node;
      nodes[left].parent = nodes[node].parent;
      nodes[node].parent = left;

      // The node's old parent should now point to its left-hand child.
      if (nodes[left].parent != NULL_NODE) {
        if (nodes[nodes[left].parent].left == node)
          nodes[nodes[left].parent].left = left;
        else {
          assert(nodes[nodes[left].parent].right == node);
          nodes[nodes[left].parent].right = left;
        }
      } else
        root = left;

      // Rotate.
      if (nodes[leftLeft].height > nodes[leftRight].height) {
        nodes[left].right = leftLeft;
        nodes[node].left = leftRight;
        nodes[leftRight].parent = node;
        nodes[node].aabb.merge(nodes[right].aabb, nodes[leftRight].aabb);
        nodes[left].aabb.merge(nodes[node].aabb, nodes[leftLeft].aabb);

        nodes[node].height =
            1 + std::max(nodes[right].height, nodes[leftRight].height);
        nodes[left].height =
            1 + std::max(nodes[node].height, nodes[leftLeft].height);
      } else {
        nodes[left].right = leftRight;
        nodes[node].left = leftLeft;
        nodes[leftLeft].parent = node;
        nodes[node].aabb.merge(nodes[right].aabb, nodes[leftLeft].aabb);
        nodes[left].aabb.merge(nodes[node].aabb, nodes[leftRight].aabb);

        nodes[node].height =
            1 + std::max(nodes[right].height, nodes[leftLeft].height);
        nodes[left].height =
            1 + std::max(nodes[node].height, nodes[leftRight].height);
      }

      return left;
    }

    return node;
  }

  [[nodiscard]] auto computeHeight() const -> size_type
  {
    return computeHeight(root);
  }

  [[nodiscard]] auto computeHeight(index_type node) const -> size_type
  {
    assert(node < nodeCapacity);

    if (nodes[node].is_leaf()) return 0;

    unsigned int height1 = computeHeight(nodes[node].left);
    unsigned int height2 = computeHeight(nodes[node].right);

    return 1 + std::max(height1, height2);
  }

  //! Assert that the sub-tree has a valid structure.
  /*! \param node
          The index of the root node.
   */
  void validateStructure(index_type node) const
  {
    if (node == NULL_NODE) return;

    if (node == root) assert(nodes[node].parent == NULL_NODE);

    unsigned int left = nodes[node].left;
    unsigned int right = nodes[node].right;

    if (nodes[node].is_leaf()) {
      assert(left == NULL_NODE);
      assert(right == NULL_NODE);
      assert(nodes[node].height == 0);
      return;
    }

    assert(left < nodeCapacity);
    assert(right < nodeCapacity);

    assert(nodes[left].parent == node);
    assert(nodes[right].parent == node);

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

    unsigned int left = nodes[node].left;
    unsigned int right = nodes[node].right;

    if (nodes[node].is_leaf()) {
      assert(left == NULL_NODE);
      assert(right == NULL_NODE);
      assert(nodes[node].height == 0);
      return;
    }

    assert(left < nodeCapacity);
    assert(right < nodeCapacity);

    int height1 = nodes[left].height;
    int height2 = nodes[right].height;
    int height = 1 + std::max(height1, height2);
    (void)height;  // Unused variable in Release build
    assert(nodes[node].height == height);

    aabb_type aabb;
    aabb.merge(nodes[left].aabb, nodes[right].aabb);

    for (unsigned int i = 0; i < dimension; i++) {
      assert(aabb.m_min[i] == nodes[node].aabb.m_min[i]);
      assert(aabb.m_max[i] == nodes[node].aabb.m_max[i]);
    }

    validateMetrics(left);
    validateMetrics(right);
  }
};

}  // namespace abby2
