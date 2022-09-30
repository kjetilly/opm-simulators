namespace Opm::cuistl::impl
{
template <typename T>
class has_communication
{
    using yes_type = char;
    using no_type = long;
    template <typename U>
    static yes_type test(decltype(&U::getCommunication));
    template <typename U>
    static no_type test(...);

public:
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes_type);
};

template <typename T>
class is_a_well_operator
{
    using yes_type = char;
    using no_type = long;
    template <typename U>
    static yes_type test(decltype(&U::addWellPressureEquations));
    template <typename U>
    static no_type test(...);

public:
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes_type);
};

template <typename T>
class has_should_call_pre
{
    using yes_type = char;
    using no_type = long;
    template <typename U>
    static yes_type test(decltype(&U::shouldCallPre));
    template <typename U>
    static no_type test(...);

public:
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes_type);
};

template <typename T>
class has_should_call_post
{
    using yes_type = char;
    using no_type = long;
    template <typename U>
    static yes_type test(decltype(&U::shouldCallPost));
    template <typename U>
    static no_type test(...);

public:
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes_type);
};

} // namespace Opm::cuistl::impl
