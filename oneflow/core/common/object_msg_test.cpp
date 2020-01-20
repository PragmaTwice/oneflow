#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

namespace test {

TEST(ObjectMsgStruct, ref_cnt) {
  class Foo final : public ObjectMsgStruct {
   public:
    Foo() = default;
  };
  Foo foo;
  foo.__InitRefCount__();
  foo.__IncreaseRefCount__();
  foo.__IncreaseRefCount__();
  ASSERT_EQ(foo.__DecreaseRefCount__(), 1);
  ASSERT_EQ(foo.__DecreaseRefCount__(), 0);
}

class TestNew final : public ObjectMsgStruct {
  BEGIN_DSS(TestNew, sizeof(ObjectMsgStruct));

  END_DSS("object_msg", TestNew);
};

TEST(ObjectMsgPtr, obj_new) { ObjectMsgPtr<TestNew>::New(); }

// clang-format off
BEGIN_OBJECT_MSG(ObjectMsgFoo)
 public:
  void __Delete__();

  OBJECT_MSG_DEFINE_FIELD(int8_t, x);
  OBJECT_MSG_DEFINE_FIELD(int32_t, foo);
  OBJECT_MSG_DEFINE_FIELD(int16_t, bar);
  OBJECT_MSG_DEFINE_FIELD(int64_t, foobar);
  OBJECT_MSG_DEFINE_RAW_PTR_FIELD(std::string*, is_deleted);
END_OBJECT_MSG(ObjectMsgFoo)
// clang-format on

void OBJECT_MSG_TYPE(ObjectMsgFoo)::__Delete__() {
  if (mutable_is_deleted()) { *mutable_is_deleted() = "deleted"; }
}

TEST(OBJECT_MSG, naive) {
  auto foo = OBJECT_MSG_PTR(ObjectMsgFoo)::New();
  foo->set_bar(9527);
  ASSERT_TRUE(foo->bar() == 9527);
}

TEST(OBJECT_MSG, __delete__) {
  std::string is_deleted;
  {
    auto foo = OBJECT_MSG_PTR(ObjectMsgFoo)::New();
    foo->set_bar(9527);
    foo->set_raw_ptr_is_deleted(&is_deleted);
    ASSERT_EQ(foo->bar(), 9527);
  }
  ASSERT_TRUE(is_deleted == "deleted");
}

// clang-format off
BEGIN_OBJECT_MSG(ObjectMsgBar)
 public:
  void __Delete__(){
    if (mutable_is_deleted()) { *mutable_is_deleted() = "bar_deleted"; }
  }
  OBJECT_MSG_DEFINE_FIELD(ObjectMsgFoo, foo);
  OBJECT_MSG_DEFINE_RAW_PTR_FIELD(std::string*, is_deleted);
END_OBJECT_MSG(ObjectMsgBar)
// clang-format on

TEST(OBJECT_MSG, nested_objects) {
  auto bar = OBJECT_MSG_PTR(ObjectMsgBar)::New();
  bar->mutable_foo()->set_bar(9527);
  ASSERT_TRUE(bar->foo().bar() == 9527);
}

TEST(OBJECT_MSG, nested_delete) {
  std::string bar_is_deleted;
  std::string is_deleted;
  {
    auto bar = OBJECT_MSG_PTR(ObjectMsgBar)::New();
    bar->set_raw_ptr_is_deleted(&bar_is_deleted);
    auto* foo = bar->mutable_foo();
    foo->set_bar(9527);
    foo->set_raw_ptr_is_deleted(&is_deleted);
    ASSERT_EQ(foo->bar(), 9527);
  }
  ASSERT_EQ(is_deleted, std::string("deleted"));
  ASSERT_EQ(bar_is_deleted, std::string("bar_deleted"));
}

}  // namespace test

}  // namespace oneflow
