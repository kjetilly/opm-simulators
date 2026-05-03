// clang-tidy suggested/example.cpp -- -std=c++17
// Extends minimum with: cppcoreguidelines-special-member-functions,
// bugprone-narrowing-conversions, cppcoreguidelines-narrowing-conversions,
// modernize-use-nullptr, bugprone-exception-escape,
// cppcoreguidelines-avoid-c-arrays + modernize-avoid-c-arrays,
// readability-function-cognitive-complexity, readability-implicit-bool-conversion.

#include <cstdio>          // OK
#include <stdio.h>         // modernize-deprecated-headers
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct BigPayload {
    std::vector<double> data{1.0, 2.0, 3.0};
    std::string name = "well";
};

class Base {
public:
    virtual ~Base() = default;
    virtual void step(double dt);
    virtual int  cells() const { return 0; }
};

// modernize-use-override: missing 'override' on both
class Derived : public Base {
public:
    void step(double dt);          // missing override
    int  cells() const { return 1; } // missing override
};

// google-explicit-constructor: single-arg ctor not explicit
class Wrapper {
public:
    Wrapper(int x) : x_(x) {}      // should be explicit
private:
    int x_;
};

// performance-unnecessary-value-param: should be const BigPayload&
double sum(BigPayload p) {
    double s = 0;
    for (auto v : p.data) s += v;
    return s;
}

// readability-make-member-function-const: getValue does not modify *this
class Counter {
    int value_ = 42;
public:
    int getValue() { return value_; }  // should be const
};

// readability-non-const-parameter: 'p' is never written through
int firstByte(char* p) {
    return static_cast<int>(p[0]);
}

// cppcoreguidelines-special-member-functions:
// user-declared destructor without copy/move operations (rule-of-5 violation)
class Buffer {
public:
    Buffer() : data_(new int[64]) {}
    ~Buffer() { delete[] data_; }   // user dtor -- needs copy/move too
private:
    int* data_;
};

// bugprone-narrowing-conversions + cppcoreguidelines-narrowing-conversions
void narrowing() {
    double d = 3.99;
    int    i = d;        // narrowing: truncates silently to 3
    long   l = 100000L;
    short  s = l;        // narrowing: may overflow
    (void)i; (void)s;
}

// modernize-use-nullptr: use 0 and NULL instead of nullptr
void legacyNull() {
    int* p = 0;          // modernize-use-nullptr
    int* q = NULL;       // modernize-use-nullptr
    if (p == 0) {}       // modernize-use-nullptr
    (void)q;
}

// bugprone-exception-escape: throwing inside a noexcept function
void riskyOperation() noexcept {
    throw std::runtime_error("unexpected");  // exception escapes noexcept
}

// cppcoreguidelines-avoid-c-arrays + modernize-avoid-c-arrays:
// raw C arrays should be std::array or std::vector
void processBuf(int buf[8]) {              // parameter decays to pointer
    int local[4] = {1, 2, 3, 4};          // local C array
    (void)buf; (void)local;
}

// readability-function-cognitive-complexity: deeply nested logic raises score
int tangle(int n) {
    int s = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            if (i % 2 == 0)
                if (j % 3 == 0)
                    for (int k = 0; k < j; ++k)
                        if (k & 1) s += k; else s -= k;
                else s += j;
            else s -= i;
    return s;
}

// readability-implicit-bool-conversion: int used where bool expected
bool isNonZero(int n) { return n; }        // implicit int -> bool
bool hasPtr(void* p) { return p; }        // implicit pointer -> bool

void demo()
{
    // cppcoreguidelines-no-malloc + cppcoreguidelines-pro-type-cstyle-cast
    int* a = (int*)malloc(sizeof(int) * 10);   // 3 warnings on one line
    free(a);

    int* b = new int(7);
    delete b;

    // cppcoreguidelines-pro-type-vararg + bugprone-suspicious-string-compare
    std::string s = "hi";
    printf("len=%zu\n", s.size());                       // vararg
    if (strcmp(s.c_str(), "hi")) { /* inverted? */ }     // suspicious-string-compare

    // bugprone-integer-division: result silently truncated to double 0.0
    double half = 1 / 2;

    // bugprone-sizeof-expression
    int arr[10];
    std::memset(arr, 0, sizeof(arr) / sizeof(int));

    // bugprone-suspicious-semicolon
    if (half > 0); { half = -half; }

    // bugprone-too-small-loop-variable
    std::vector<int> big(70000);
    for (short i = 0; i < big.size(); ++i) { big[i] = i; }

    // bugprone-use-after-move
    std::string a1 = "abc";
    std::string a2 = std::move(a1);
    (void)a1.size();                                     // use-after-move

    // performance-move-const-arg: std::move on const has no effect
    const std::string c = "x";
    std::string d = std::move(c);

    // cppcoreguidelines-slicing
    Derived der;
    Base base = der;                                     // slicing

    // performance-for-range-copy
    std::vector<std::string> names{"a","b","c"};
    for (std::string n : names) { (void)n.size(); }      // copy each iter

    // bugprone-unused-return-value
    s.empty();

    // unused warnings:
    (void)firstByte(nullptr);
    (void)sum(BigPayload{});
    narrowing();
    legacyNull();
}
