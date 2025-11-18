#ifndef __KDTREE_H__
#define __KDTREE_H__

#include <algorithm>
#include <limits>
#include <vector>

#ifdef __GNUC__
#include <queue>
#else
#include <queue>
#endif

#include "rdr/accel.h"
#include "rdr/rdr.h"

RDR_NAMESPACE_BEGIN

template <typename PointType_, typename DataType_> struct KDNode {
  using PointType = PointType_;
  using DataType = DataType_;
  using IndexType = int;

  constexpr static int INVALID_INDEX = -1;

  PointType position;
  DataType data{};
  IndexType left{INVALID_INDEX}, right{INVALID_INDEX};
  int axis{INVALID_INDEX}; // the id of split axis
  bool is_leaf{false};

  RDR_FORCEINLINE KDNode() = default;
  RDR_FORCEINLINE KDNode(const PointType &position) : position(position) {}
  RDR_FORCEINLINE KDNode(const PointType &position, const DataType &data)
      : position(position), data(data) {}

  RDR_FORCEINLINE const PointType &getPosition() const { return position; }
  RDR_FORCEINLINE PointType &getPosition() { return position; }

  RDR_FORCEINLINE const DataType &getData() const { return data; }
  RDR_FORCEINLINE DataType &getData() { return data; }

  RDR_FORCEINLINE const IndexType &getLeftIndex() const { return left; }
  RDR_FORCEINLINE IndexType &getLeftIndex() { return left; }

  RDR_FORCEINLINE const IndexType &getRightIndex() const { return right; }
  RDR_FORCEINLINE IndexType &getRightIndex() { return right; }

  RDR_FORCEINLINE int getAxis() const { return axis; }
  RDR_FORCEINLINE int &getAxis() { return axis; }

  RDR_FORCEINLINE bool isLeaf() const { return is_leaf; }
  RDR_FORCEINLINE void setLeaf(bool value) { is_leaf = value; }

  RDR_FORCEINLINE bool inLeft(const PointType &other) const {
    assert(axis != -1 && !is_leaf);
    return other[axis] <= getPosition()[axis];
  }

  RDR_FORCEINLINE bool inLeft(const KDNode &other) const {
    return inLeft(other.getPosition());
  }
};

/**
 * @brief A simple and incomplete implementation of KD-Tree specifically for
 * Photon Mapping. KD-Tree does not support dynamic node add: once the tree is
 * built, no modification is possible without re-build.
 *
 * @tparam _NodeType
 * @tparam _AABBType
 */
template <typename NodeType_, typename AABBType_> class KDTree final {
public:
  using NodeType = NodeType_;
  using AABBType = AABBType_;
  using PointType = typename NodeType::PointType;
  using DataType = typename NodeType::DataType;
  using IndexType = typename NodeType::IndexType;
  using IteratorType = typename vector<NodeType>::iterator;
  using NodeCallbackType = std::function<void(const NodeType &)>;
  using IndexCallbackType = std::function<void(const IndexType &)>;

  // The number of dimensions
  constexpr static int K = vec_type<PointType>::size;
  constexpr static int INVALID_INDEX = NodeType::INVALID_INDEX;

  KDTree() = default;
  KDTree(KDTree &&) = delete;
  KDTree &operator=(KDTree &&) = delete;
  KDTree(const KDTree &) = delete;
  KDTree &operator=(const KDTree &) = delete;
  ~KDTree() = default;

  /// std-like interface
  size_t size() const { return nodes.size(); }
  const NodeType &operator[](size_t i) const { return nodes[i]; }
  NodeType &operator[](size_t i) { return nodes[i]; }

  /// Reset the state of the KD-Tree
  void clear() {
    // clear the nodes without affecting its capacity hopefully
    nodes.resize(0);
    aabb = AABBType();
    root_index = INVALID_INDEX;
  }

  void push_back(const NodeType &node) { // NOLINT
    nodes.push_back(node);
    aabb = AABBType(aabb, node.getPosition());
  }

  void push_back(NodeType &&node) { // NOLINT
    nodes.push_back(std::move(node));
    aabb = AABBType(aabb, node.getPosition());
  }

  const vector<NodeType> &getNodes() const { return nodes; }
  const AABBType &getAABB() const { return aabb; }

  /// Build the tree
  void build() { root_index = build(0, 0, size()); }

  /**
   * @brief Perform nearest neighbor search within the tree around the reference
   * point, and return the **index** of the nearest node. One should further
   * fetch the node with this index.
   *
   * @param point The reference point
   * @param min_sqr_distance The minimum squared distance between the reference
   * @return IndexType
   */
  IndexType nearestNeighborSearch(const PointType &point,
                                  Float &min_sqr_distance) const {
    IndexType min_index = root_index;
    nearestNeighborSearch(root_index, point, min_sqr_distance, min_index);
    return min_index;
  }

  /// @see nearestNeighborSearch
  IndexType nearestNeighborSearch(const PointType &point) const {
    IndexType min_index = root_index;
    Float min_sqr_distance = std::numeric_limits<Float>::max();
    return nearestNeighborSearch(point, min_sqr_distance);
  }

  /**
   * @brief Perform fixed-radius search within the tree around the reference and
   * invoke the callback with node index. The callback is guaranteed to be
   * invoked in a single thread thus you don't need to care about thread safety.
   *
   * @param point The reference point
   * @param max_distance The maximum distance to look for
   * @param callback The callback function to be invoked
   */
  void fixedRadiusSearch(const PointType &point, Float max_distance,
                         IndexCallbackType callback) const {
    fixedRadiusSearch(root_index, point, max_distance * max_distance, callback);
  }

  /// @see fixedRadiusSearch. The difference is, this function will invoke the
  /// callback with node instead of index, so you might not change the node.
  void fixedRadiusSearch(const PointType &point, Float max_distance,
                         NodeCallbackType node_callback) const {
    fixedRadiusSearch(root_index, point, max_distance * max_distance,
                      [this, node_callback](const IndexType &index) {
                        node_callback(nodes[index]);
                      });
  }

  /**
   * @brief Perform k-nearest neighbor search within the tree around the
   * reference point. That is, invoke the callback function with the index of
   * the k-nearest points.
   *
   * @param point The reference point
   * @param k The number of nearest neighbors to look for
   * @param callback The callback function to be invoked with node index
   */
  void kNearestNeighborSearch(const PointType &point, size_t k,
                              IndexCallbackType callback) const {
    // 使用 lambda 定义比较器
    auto comp = [this, point](const IndexType &a, const IndexType &b) {
      return SquareNorm(getIterator(a)->getPosition() - point) <
             SquareNorm(getIterator(b)->getPosition() - point);
    };

    // 使用 decltype 获取 lambda 的类型
    std::priority_queue<IndexType, vector<IndexType>, decltype(comp)> heap(
        comp);

    kNearestNeighborSearch(root_index, point, k, heap);

    // now that the heap is filled
    // invoke in inverse-order
    while (!heap.empty()) {
      callback(heap.top());
      heap.pop();
    }
  }

  /// @see kNearestNeighborSearch. The difference is, this function will invoke
  /// the callback with node instead of index, so you might not change the node.
  void kNearestNeighborSearch(const PointType &point, size_t k,
                              NodeCallbackType callback) const {
    kNearestNeighborSearch(point, k, [this, callback](const IndexType &index) {
      callback(nodes[index]);
    });
  };

private:
  std::deque<NodeType> nodes{};
  AABBType aabb{};
  IndexType root_index{};

  RDR_FORCEINLINE decltype(auto) getIterator(const IndexType &index) {
    return nodes.begin() + index;
  }

  RDR_FORCEINLINE decltype(auto) getIterator(const IndexType &index) const {
    return nodes.begin() + index;
  }

  /// Build the tree with a given max depth
  IndexType build(int depth, IndexType start, IndexType end) {
    const int axis = depth % K;
    const int count = end - start;

    // balanced KD-Tree
    IndexType split = start + count / 2;
    if (count <= 0)
      return INVALID_INDEX;

    assert(count >= 1);
    if (count == 1) {
      // this node is the leaf
      assert(start == split);
      getIterator(split)->setLeaf(true);
      return split;
    }

    // Split elements into two parts
    std::nth_element(getIterator(start), getIterator(split), getIterator(end),
                     [&](const auto &a, const auto &b) {
                       return a.getPosition()[axis] < b.getPosition()[axis];
                     });

    auto iter = getIterator(split);
    iter->getAxis() = axis;

    // Note that the current point is not considered here
    auto left = build(depth + 1, start, split);    // LT
    auto right = build(depth + 1, split + 1, end); // GT
    iter->getLeftIndex() = left;
    iter->getRightIndex() = right;

    return split;
  }

  /**
   * @brief View node_index as the root, recursively calculate the nearest
   * neighbor under this subtree(including itself).
   *
   * @return IndexType The index of the nearest neighbor under this subtree.
   */
  void nearestNeighborSearch(const IndexType &node_index,
                             const PointType &point, Float &min_sqr_distance,
                             IndexType &min_index) const {
    if (node_index == INVALID_INDEX)
      return;
    auto node = getIterator(node_index);

    Float sqr_distance = SquareNorm(point - node->getPosition());
    if (sqr_distance < min_sqr_distance) {
      min_index = node_index;
      min_sqr_distance = sqr_distance;
    }

    if (node->isLeaf()) {
      return;
    }

    // TopDown phase
    auto first_index = node->getLeftIndex();
    auto second_index = node->getRightIndex();
    if (!node->inLeft(point))
      std::swap(first_index, second_index);
    nearestNeighborSearch(first_index, point, min_sqr_distance, min_index);

    // rewind phase: consider the other directio
    const int axis = node->getAxis();
    sqr_distance = node->getPosition()[axis] - point[axis];
    sqr_distance = sqr_distance * sqr_distance;

    // if the distance between point and the plane is already larger, then
    // there's not need to traverse
    if (sqr_distance <= min_sqr_distance)
      nearestNeighborSearch(second_index, point, min_sqr_distance, min_index);
  }

  /**
   * @brief View node_index as the root, recursively traverse all the nodes
   * inside this tree with maximum square distance less than max_sqr_distance
   * while calling the callback function. Notice that this function does not
   * support write-back.
   */
  void fixedRadiusSearch(const IndexType &node_index, const PointType &point,
                         Float max_sqr_distance,
                         IndexCallbackType callback) const {
    if (node_index == INVALID_INDEX)
      return;
    auto node = getIterator(node_index);

    Float sqr_distance = SquareNorm(point - node->getPosition());
    // process the currect node
    if (sqr_distance < max_sqr_distance) {
      callback(node_index);
    }

    if (node->isLeaf()) {
      return;
    }

    // topdown
    auto first_index = node->getLeftIndex();
    auto second_index = node->getRightIndex();

    if (!node->inLeft(point))
      std::swap(first_index, second_index);
    fixedRadiusSearch(first_index, point, max_sqr_distance, callback);

    // rewind, perform pruning
    const int axis = node->getAxis();
    sqr_distance = node->getPosition()[axis] - point[axis];
    sqr_distance = sqr_distance * sqr_distance;

    if (sqr_distance <= max_sqr_distance)
      fixedRadiusSearch(second_index, point, max_sqr_distance, callback);
  }

  /**
   * @brief View node_index as the root, recursively traverse all the nodes
   * inside this tree while maintaining all the k nodes with smaller distance */
  template <typename HeapType>
  void kNearestNeighborSearch(const IndexType &node_index,
                              const PointType &point, size_t k,
                              HeapType &heap) const {
    if (node_index == INVALID_INDEX)
      return;
    auto node = getIterator(node_index);

    if (heap.empty() || heap.size() < k ||
        SquareNorm(point - node->getPosition()) <
            SquareNorm(point - getIterator(heap.top())->getPosition())) {
      heap.push(node_index);
      if (heap.size() > k)
        heap.pop();
    }

    if (node->isLeaf()) {
      return;
    }

    // topdown
    auto first_index = node->getLeftIndex();
    auto second_index = node->getRightIndex();

    if (!node->inLeft(point))
      std::swap(first_index, second_index);
    kNearestNeighborSearch(first_index, point, k, heap);

    if (!heap.empty()) {
      // rewind, perform pruning
      const int axis = node->getAxis();
      Float sqr_distance = node->getPosition()[axis] - point[axis];
      sqr_distance = sqr_distance * sqr_distance;

      // If the distance is less than the currect max distance, then we need
      // to traverse the next tree
      assert(!heap.empty());
      if (sqr_distance <=
          SquareNorm(getIterator(heap.top())->getPosition() - point))
        kNearestNeighborSearch(second_index, point, k, heap);
    } else {
      kNearestNeighborSearch(second_index, point, k, heap);
    }
  }
};

template <typename DataType>
using KDTree2 = KDTree<KDNode<Vec2f, DataType>, TAABB<Vec2f>>;

template <typename DataType>
using KDTree3 = KDTree<KDNode<Vec3f, DataType>, TAABB<Vec3f>>;

RDR_NAMESPACE_END

#endif
