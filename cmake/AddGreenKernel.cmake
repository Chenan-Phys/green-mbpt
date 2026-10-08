
function(add_green_kernel CUSTOM_KERNELS_IN)
    set(GREEN_GPU_REVISION "477dfdbc5a6310fc70ca42ba340fa5af18c37554" CACHE STRING "Coordinated THC GPU revision")
    set(CUSTOM_KERNELS_TMP "${CUSTOM_KERNELS_IN}")
    set(CUSTOM_KERNELS_LST "")
    foreach(KERNEL ${CUSTOM_KERNELS_TMP})
        string(REGEX MATCH "([^/]+)/?$" KERNEL_NAME ${KERNEL})
        if(KERNEL_NAME STREQUAL "green-gpu.git")
            set(KERNEL_NAME "green-gpu")
        endif()
        message("Adding kernel ${KERNEL_NAME} ${KERNEL}")
        if(NOT DEFINED KERNEL_NAME)
            message(FATAL_ERROR "Can not extract kernel name")
        endif()

        Include(FetchContent)

        set(KERNEL_REVISION "${GREEN_RELEASE}")
        set(KERNEL_SOURCE "${KERNEL}")
        if(KERNEL_NAME STREQUAL "green-gpu")
            set(KERNEL_REVISION "${GREEN_GPU_REVISION}")
            if(KERNEL STREQUAL "https://github.com/Green-Phys/green-gpu" OR KERNEL STREQUAL "https://github.com/Green-Phys/green-gpu.git")
                set(KERNEL_SOURCE "https://github.com/Chenan-Phys/green-gpu.git")
            endif()
        endif()
        FetchContent_Declare(
            ${KERNEL_NAME}
            GIT_REPOSITORY ${KERNEL_SOURCE}
            GIT_TAG ${KERNEL_REVISION}
        )

        FetchContent_MakeAvailable(${KERNEL_NAME})
        list(APPEND CUSTOM_KERNELS_LST ${KERNEL})
    endforeach()
    set(CUSTOM_KERNELS "${CUSTOM_KERNELS_LST}" PARENT_SCOPE)
endfunction()
