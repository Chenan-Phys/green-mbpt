
function(add_green_kernel CUSTOM_KERNELS_IN)
    set(GREEN_GPU_REVISION "8a529340bab7f7e281033d64311edb2962b03ef9" CACHE STRING "Coordinated THC GPU revision")
    set(CUSTOM_KERNELS_TMP "${CUSTOM_KERNELS_IN}")
    set(CUSTOM_KERNELS_LST "")
    foreach(KERNEL ${CUSTOM_KERNELS_TMP})
        string(REGEX MATCH "([^/]+)/?$" KERNEL_NAME ${KERNEL})
        message("Adding kernel ${KERNEL_NAME} ${KERNEL}")
        if(NOT DEFINED KERNEL_NAME)
            message(FATAL_ERROR "Can not extract kernel name")
        endif()

        Include(FetchContent)

        set(KERNEL_REVISION "${GREEN_RELEASE}")
        if(KERNEL_NAME STREQUAL "green-gpu")
            set(KERNEL_REVISION "${GREEN_GPU_REVISION}")
        endif()
        FetchContent_Declare(
            ${KERNEL_NAME}
            GIT_REPOSITORY ${KERNEL}
            GIT_TAG ${KERNEL_REVISION}
        )

        FetchContent_MakeAvailable(${KERNEL_NAME})
        list(APPEND CUSTOM_KERNELS_LST ${KERNEL})
    endforeach()
    set(CUSTOM_KERNELS "${CUSTOM_KERNELS_LST}" PARENT_SCOPE)
endfunction()
