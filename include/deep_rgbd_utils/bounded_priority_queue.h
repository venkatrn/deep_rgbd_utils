#pragma once

// Based on http://codereview.stackexchange.com/questions/68126/limited-memory-priority-queue

#include <array>
#include <algorithm>

template<typename T, std::size_t N>
class bounded_priority_queue {
 public:
  bounded_priority_queue() {}
  bounded_priority_queue(const bounded_priority_queue& other) :
    items(other.items),
    next(items.begin() + other.size()) {}

  void push(const T& item) {
    if (next != items.end()) {
      *next = item;
      ++next;
      std::push_heap(items.begin(), next);
    } else {
      std::sort_heap(items.begin(), items.end());

      if (items.front() < item) {
        items[0] = item;
      }

      std::make_heap(items.begin(), items.end());
    }
  }

  const T &top() const {
    return items.front();
  }

  void pop() {
    std::pop_heap(items.begin(), next);
    --next;
  }

  bool empty() const {
    return size() == 0;
  }

  std::size_t size() const {
    return next - items.begin();
  }

 private:
  std::array<T, N> items = {};
  T *next = items.begin();
};
