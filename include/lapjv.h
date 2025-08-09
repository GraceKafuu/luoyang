/*
 * lapjv.h
 * 线性分配问题的Jonker-Volgenant算法头文件
 */
#pragma once

#ifndef LAPJV_H
#define LAPJV_H

// 定义一个大数值，用于初始化或特殊标记
#define LARGE 1000000

// 布尔值定义
#if !defined TRUE
#define TRUE 1
#endif
#if !defined FALSE
#define FALSE 0
#endif

// 动态内存分配宏，如果分配失败则返回-1
#define NEW(x, t, n) if ((x = (t *)malloc(sizeof(t) * (n))) == 0) { return -1; }
// 安全释放内存宏，释放后将指针置为NULL
#define FREE(x) if (x != 0) { free(x); x = 0; }
// 交换两个索引值的宏
#define SWAP_INDICES(a, b) { int_t _temp_index = a; a = b; b = _temp_index; }

// 调试相关的宏定义
#if 0
#include <assert.h>
#define ASSERT(cond) assert(cond)
#define PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
// 打印代价数组的宏
#define PRINT_COST_ARRAY(a, n) \
    while (1) { \
        printf(#a" = ["); \
        if ((n) > 0) { \
            printf("%f", (a)[0]); \
            for (uint_t j = 1; j < n; j++) { \
                printf(", %f", (a)[j]); \
            } \
        } \
        printf("]\n"); \
        break; \
    }
// 打印索引数组的宏
#define PRINT_INDEX_ARRAY(a, n) \
    while (1) { \
        printf(#a" = ["); \
        if ((n) > 0) { \
            printf("%d", (a)[0]); \
            for (uint_t j = 1; j < n; j++) { \
                printf(", %d", (a)[j]); \
            } \
        } \
        printf("]\n"); \
        break; \
    }
#else
// 空定义，禁用调试输出
#define ASSERT(cond)
#define PRINTF(fmt, ...)
#define PRINT_COST_ARRAY(a, n)
#define PRINT_INDEX_ARRAY(a, n)
#endif

// 类型别名定义，提高代码可读性
typedef signed int int_t;          // 有符号整型
typedef unsigned int uint_t;       // 无符号整型
typedef double cost_t;             // 代价类型
typedef char boolean;              // 布尔类型
// 浮点数类型枚举，表示不同的浮点精度选项
typedef enum fp_t { FP_1 = 1, FP_2 = 2, FP_DYNAMIC = 3 } fp_t;

// LAPJV算法核心函数声明
// 参数:
//   n: 问题规模(矩阵大小)
//   cost: 代价矩阵
//   x: 输出参数，表示行分配结果
//   y: 输出参数，表示列分配结果
// 返回值: 错误代码，0表示成功
extern int_t lapjv_internal(
	const uint_t n, cost_t *cost[],
	int_t *x, int_t *y);

#endif // LAPJV_H