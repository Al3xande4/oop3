cmake_minimum_required(VERSION 3.24)
project(sem3-oop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

# Санитайзеры только в Debug: покрывают ошибки памяти и неопределённое поведение.
# Переопределяется через -DSANITIZE=OFF, если инструменты недоступны.
option(SANITIZE "Enable AddressSanitizer and UndefinedBehaviorSanitizer (Debug)" ON)
if(SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()


add_subdirectory(src)

enable_testing()
add_subdirectory(tests)