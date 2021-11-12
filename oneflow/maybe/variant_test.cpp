/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <gtest/gtest.h>
#include <memory>
#include "oneflow/maybe/variant.h"

using namespace oneflow::maybe;

TEST(Variant, Basics) {
  Variant<int, float> a, b(1), c(1.2f), d(InPlaceType<int>, 'a'), e(InPlaceType<float>, 6.66);
  ASSERT_TRUE(a.Is<int>());
  ASSERT_EQ(a.Get<int>(), 0);
  ASSERT_TRUE(b.Is<int>());
  ASSERT_EQ(b.Get<int>(), 1);
  ASSERT_TRUE(c.Is<float>());
  ASSERT_EQ(c.Get<float>(), 1.2f);
  ASSERT_TRUE(d.Is<int>());
  ASSERT_EQ(d.Get<int>(), 'a');
  ASSERT_TRUE(e.Is<float>());
  ASSERT_FLOAT_EQ(e.Get<float>(), 6.66);

  Variant<int, float> f(b), g(c), h(InPlaceIndex<1>, 2.33), i(InPlaceIndex<0>, 2.33);
  ASSERT_TRUE(f.Is<int>());
  ASSERT_EQ(f.Get<int>(), 1);
  ASSERT_TRUE(g.Is<float>());
  ASSERT_EQ(g.Get<float>(), 1.2f);
  ASSERT_TRUE(h.Is<float>());
  ASSERT_FLOAT_EQ(h.Get<float>(), 2.33);
  ASSERT_TRUE(i.Is<int>());
  ASSERT_EQ(i.Get<int>(), 2);

  a = 1;
  ASSERT_TRUE(a.Is<int>());
  ASSERT_EQ(a.Get<int>(), 1);

  a = 1.3f;
  ASSERT_TRUE(a.Is<float>());
  ASSERT_EQ(a.Get<float>(), 1.3f);

  a = b;
  ASSERT_TRUE(a.Is<int>());
  ASSERT_EQ(a.Get<int>(), 1);

  a = c;
  ASSERT_TRUE(a.Is<float>());
  ASSERT_EQ(a.Get<float>(), 1.2f);

  ASSERT_EQ((b.visit<Variant<int, float>>([](auto&& x) { return x + 1; })),
            (Variant<int, float>(2)));
  ASSERT_EQ((c.visit<Variant<int, float>>([](auto&& x) { return x + 1; })),
            (Variant<int, float>(2.2f)));

  ASSERT_EQ(a.Emplace<1>(1.3f), 1.3f);
  ASSERT_TRUE(a.Is<float>());
  ASSERT_EQ(a.Get<1>(), 1.3f);

  ASSERT_EQ(a.Emplace<0>(233), 233);
  ASSERT_TRUE(a.Is<int>());
  ASSERT_EQ(a.Get<0>(), 233);
}

TEST(Variant, NonPOD) {
  Variant<bool, std::shared_ptr<int>> a;
  ASSERT_TRUE(a.Is<bool>());
  ASSERT_EQ(a.Get<bool>(), false);

  a = true;
  ASSERT_TRUE(a.Is<bool>());
  ASSERT_EQ(a.Get<bool>(), true);

  a = std::make_shared<int>(233);
  ASSERT_EQ(a.Index(), 1);
  ASSERT_EQ(*a.Get<1>(), 233);
  ASSERT_EQ(a.Get<1>().use_count(), 1);

  {
    Variant<bool, std::shared_ptr<int>> b = a;
    ASSERT_EQ(b.Index(), 1);
    ASSERT_EQ(*b.Get<1>(), 233);
    ASSERT_EQ(a.Get<1>().use_count(), 2);
    *b.Get<1>() = 234;
  }

  ASSERT_EQ(a.Get<1>().use_count(), 1);
  ASSERT_EQ(*a.Get<1>(), 234);

  Variant<bool, std::shared_ptr<int>> b = std::move(a);
  ASSERT_EQ(b.Get<1>().use_count(), 1);
  ASSERT_EQ(*b.Get<1>(), 234);

  Variant<bool, std::shared_ptr<int>> c = b;
  ASSERT_EQ(c.Get<1>().use_count(), 2);
  ASSERT_EQ(b, c);

  b = true;
  ASSERT_EQ(c.Get<1>().use_count(), 1);

  ASSERT_NE(b, c);
}

TEST(Variant, Optional) {
  OptionalVariant<int, const char*> a, b(NullOpt), c(a);

  const char* hello = "hello";

  std::size_t hash = 0, hash2 = 1, hash3 = 2;
  HashCombine(hash, NullOpt);
  HashCombine(hash2, 1);
  HashCombine(hash3, hello);

  ASSERT_TRUE(a == NullOpt);
  ASSERT_EQ(std::hash<decltype(a)>()(a), hash);

  a = 1;
  ASSERT_EQ(a, 1);
  ASSERT_EQ(std::hash<decltype(a)>()(a), hash2);

  a = NullOpt;
  ASSERT_EQ(a, NullOpt);
  ASSERT_EQ(std::hash<decltype(a)>()(a), hash);

  a = hello;
  ASSERT_EQ(a, hello);
  ASSERT_EQ(std::hash<decltype(a)>()(a), hash3);

  ASSERT_EQ(b, NullOpt);
  ASSERT_EQ(c, NullOpt);
  ASSERT_NE(a, b);
}
