#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lapjv.h"

/** 列简化和简化转移，用于稠密代价矩阵.
 *  这是JV算法的第一步，通过列简化来寻找初始解
 *  \param n 问题规模(矩阵大小)
 *  \param cost 代价矩阵
 *  \param free_rows 返回未分配的行索引
 *  \param x 行分配结果数组
 *  \param y 列分配结果数组
 *  \param v 列最小值数组
 *  \return 未分配的行数
 */
int_t _ccrrt_dense(const uint_t n, cost_t *cost[],
	int_t *free_rows, int_t *x, int_t *y, cost_t *v)
{
	int_t n_free_rows;
	boolean *unique;

	// 初始化
	for (uint_t i = 0; i < n; i++) {
		x[i] = -1;         // 行分配初始化为-1，表示未分配
		v[i] = LARGE;      // 列最小值初始化为大数
		y[i] = 0;          // 列分配初始化为0
	}
	
	// 对每一列，找到该列的最小元素和对应的行索引
	for (uint_t i = 0; i < n; i++) {
		for (uint_t j = 0; j < n; j++) {
			const cost_t c = cost[i][j];
			if (c < v[j]) {
				v[j] = c;      // 更新列最小值
				y[j] = i;      // 更新该列最小值所在的行
			}
			PRINTF("i=%d, j=%d, c[i,j]=%f, v[j]=%f y[j]=%d\n", i, j, c, v[j], y[j]);
		}
	}
	PRINT_COST_ARRAY(v, n);
	PRINT_INDEX_ARRAY(y, n);
	
	// 为唯一最小值的行分配列
	NEW(unique, boolean, n);
	memset(unique, TRUE, n);
	{
		int_t j = n;
		do {
			j--;
			const int_t i = y[j];     // 第j列的最小元素所在行
			if (x[i] < 0) {
				x[i] = j;             // 如果该行未分配，则分配第j列给它
			}
			else {
				unique[i] = FALSE;    // 该行不是唯一的，标记为FALSE
				y[j] = -1;            // 该列暂时不分配
			}
		} while (j > 0);
	}
	
	// 处理未分配的行和非唯一分配的行
	n_free_rows = 0;
	for (uint_t i = 0; i < n; i++) {
		if (x[i] < 0) {
			// 收集未分配的行
			free_rows[n_free_rows++] = i;
		}
		else if (unique[i]) {
			// 对于有唯一最小值的行，更新其对应的列值
			const int_t j = x[i];
			cost_t min = LARGE;
			// 计算该行除已分配列外其他列的最小代价
			for (uint_t j2 = 0; j2 < n; j2++) {
				if (j2 == (uint_t)j) {
					continue;
				}
				const cost_t c = cost[i][j2] - v[j2];
				if (c < min) {
					min = c;
				}
			}
			PRINTF("v[%d] = %f - %f\n", j, v[j], min);
			v[j] -= min;  // 调整列值
		}
	}
	FREE(unique);
	return n_free_rows;
}


/** 稠密代价矩阵的增广行简化.
 *  这是JV算法的第二步，通过行简化进一步优化解
 *  \param n 问题规模(矩阵大小)
 *  \param cost 代价矩阵
 *  \param n_free_rows 未分配行的数量
 *  \param free_rows 未分配行数组
 *  \param x 行分配结果数组
 *  \param y 列分配结果数组
 *  \param v 列最小值数组
 *  \return 新的未分配行数
 */
int_t _carr_dense(
	const uint_t n, cost_t *cost[],
	const uint_t n_free_rows,
	int_t *free_rows, int_t *x, int_t *y, cost_t *v)
{
	uint_t current = 0;
	int_t new_free_rows = 0;
	uint_t rr_cnt = 0;
	PRINT_INDEX_ARRAY(x, n);
	PRINT_INDEX_ARRAY(y, n);
	PRINT_COST_ARRAY(v, n);
	PRINT_INDEX_ARRAY(free_rows, n_free_rows);
	
	// 处理所有未分配的行
	while (current < n_free_rows) {
		int_t i0;
		int_t j1, j2;
		cost_t v1, v2, v1_new;
		boolean v1_lowers;

		rr_cnt++;
		PRINTF("current = %d rr_cnt = %d\n", current, rr_cnt);
		const int_t free_i = free_rows[current++];  // 取出一个未分配的行
		// 寻找该行的两个最小代价列
		j1 = 0;
		v1 = cost[free_i][0] - v[0];  // 第一最小值
		j2 = -1;
		v2 = LARGE;                   // 第二最小值
		for (uint_t j = 1; j < n; j++) {
			PRINTF("%d = %f %d = %f\n", j1, v1, j2, v2);
			const cost_t c = cost[free_i][j] - v[j];
			if (c < v2) {
				if (c >= v1) {
					v2 = c;
					j2 = j;
				}
				else {
					v2 = v1;
					v1 = c;
					j2 = j1;
					j1 = j;
				}
			}
		}
		i0 = y[j1];  // 当前j1列分配的行
		v1_new = v[j1] - (v2 - v1);  // 计算新的v值
		v1_lowers = v1_new < v[j1];  // 判断v值是否降低
		PRINTF("%d %d 1=%d,%f 2=%d,%f v1'=%f(%d,%g) \n", free_i, i0, j1, v1, j2, v2, v1_new, v1_lowers, v[j1] - v1_new);
		
		// 根据条件更新分配
		if (rr_cnt < current * n) {
			if (v1_lowers) {
				v[j1] = v1_new;
			}
			else if (i0 >= 0 && j2 >= 0) {
				j1 = j2;
				i0 = y[j2];
			}
			if (i0 >= 0) {
				if (v1_lowers) {
					free_rows[--current] = i0;  // 将原分配行放回未分配列表
				}
				else {
					free_rows[new_free_rows++] = i0;  // 将原分配行加入新的未分配列表
				}
			}
		}
		else {
			PRINTF("rr_cnt=%d >= %d (current=%d * n=%d)\n", rr_cnt, current * n, current, n);
			if (i0 >= 0) {
				free_rows[new_free_rows++] = i0;
			}
		}
		x[free_i] = j1;  // 更新分配结果
		y[j1] = free_i;
	}
	return new_free_rows;
}


/** 查找具有最小d[j]值的列，并将它们放入SCAN列表中.
 *  在最短路径算法中使用，用于找到当前最有利的列
 *  \param n 问题规模
 *  \param lo 起始索引
 *  \param d 距离数组
 *  \param cols 列索引数组
 *  \param y 列分配数组
 *  \return 最小值元素的结束位置
 */
uint_t _find_dense(const uint_t n, uint_t lo, cost_t *d, int_t *cols, int_t *y)
{
	uint_t hi = lo + 1;
	cost_t mind = d[cols[lo]];  // 当前最小距离
	// 遍历剩余列，找到具有最小距离的列
	for (uint_t k = hi; k < n; k++) {
		int_t j = cols[k];
		if (d[j] <= mind) {
			if (d[j] < mind) {
				hi = lo;        // 重置hi
				mind = d[j];    // 更新最小距离
			}
			// 交换列索引，将具有相同最小距离的列放到前面
			cols[k] = cols[hi];
			cols[hi++] = j;
		}
	}
	return hi;
}


/** 扫描TODO列表中的所有列，从SCAN列表中的任意列开始，尝试减少TODO列的d值.
 *  这是Dijkstra最短路径算法的修改版本中的扫描步骤
 *  \param n 问题规模
 *  \param cost 代价矩阵
 *  \param plo SCAN列表的起始位置指针
 *  \param phi SCAN列表的结束位置指针
 *  \param d 距离数组
 *  \param cols 列索引数组
 *  \param pred 前驱数组
 *  \param y 列分配数组
 *  \param v 列最小值数组
 *  \return 找到的未分配列索引，如果未找到则返回-1
 */
int_t _scan_dense(const uint_t n, cost_t *cost[],
	uint_t *plo, uint_t*phi,
	cost_t *d, int_t *cols, int_t *pred,
	int_t *y, cost_t *v)
{
	uint_t lo = *plo;
	uint_t hi = *phi;
	cost_t h, cred_ij;

	// 扫描SCAN列表中的所有列
	while (lo != hi) {
		int_t j = cols[lo++];
		const int_t i = y[j];           // 第j列当前分配的行
		const cost_t mind = d[j];       // 第j列的当前距离
		h = cost[i][j] - v[j] - mind;   // 计算调整值
		PRINTF("i=%d j=%d h=%f\n", i, j, h);
		
		// 对所有TODO列表中的列进行处理
		for (uint_t k = hi; k < n; k++) {
			j = cols[k];
			cred_ij = cost[i][j] - v[j] - h;  // 计算新的距离
			if (cred_ij < d[j]) {
				d[j] = cred_ij;      // 更新距离
				pred[j] = i;         // 更新前驱
				if (cred_ij == mind) {
					if (y[j] < 0) {
						return j;    // 找到未分配列，返回
					}
					// 将该列加入SCAN列表
					cols[k] = cols[hi];
					cols[hi++] = j;
				}
			}
		}
	}
	*plo = lo;
	*phi = hi;
	return -1;  // 未找到未分配列
}


/** 修改的Dijkstra最短路径算法的单次迭代，如JV论文中所述.
 *
 * 这是稠密矩阵版本.
 *
 * \param n 问题规模
 * \param cost 代价矩阵
 * \param start_i 起始行索引
 * \param y 列分配数组
 * \param v 列最小值数组
 * \param pred 前驱数组
 * \return 最近的未分配列索引.
 */
int_t find_path_dense(
	const uint_t n, cost_t *cost[],
	const int_t start_i,
	int_t *y, cost_t *v,
	int_t *pred)
{
	uint_t lo = 0, hi = 0;
	int_t final_j = -1;
	uint_t n_ready = 0;
	int_t *cols;
	cost_t *d;

	NEW(cols, int_t, n);
	NEW(d, cost_t, n);

	// 初始化
	for (uint_t i = 0; i < n; i++) {
		cols[i] = i;                    // 初始时所有列都在TODO列表中
		pred[i] = start_i;              // 所有列的前驱都设为起始行
		d[i] = cost[start_i][i] - v[i]; // 初始化距离
	}
	PRINT_COST_ARRAY(d, n);
	
	// 循环直到找到未分配列
	while (final_j == -1) {
		// SCAN列表为空
		if (lo == hi) {
			PRINTF("%d..%d -> find\n", lo, hi);
			n_ready = lo;
			// 查找具有最小距离的列
			hi = _find_dense(n, lo, d, cols, y);
			PRINTF("check %d..%d\n", lo, hi);
			PRINT_INDEX_ARRAY(cols, n);
			// 检查这些列中是否有未分配的
			for (uint_t k = lo; k < hi; k++) {
				const int_t j = cols[k];
				if (y[j] < 0) {
					final_j = j;  // 找到未分配列
				}
			}
		}
		if (final_j == -1) {
			PRINTF("%d..%d -> scan\n", lo, hi);
			// 扫描处理
			final_j = _scan_dense(
				n, cost, &lo, &hi, d, cols, pred, y, v);
			PRINT_COST_ARRAY(d, n);
			PRINT_INDEX_ARRAY(cols, n);
			PRINT_INDEX_ARRAY(pred, n);
		}
	}

	PRINTF("found final_j=%d\n", final_j);
	PRINT_INDEX_ARRAY(cols, n);
	{
		const cost_t mind = d[cols[lo]];  // 最小距离
		// 更新v值
		for (uint_t k = 0; k < n_ready; k++) {
			const int_t j = cols[k];
			v[j] += d[j] - mind;
		}
	}

	FREE(cols);
	FREE(d);

	return final_j;
}


/** 稠密代价矩阵的增广.
 *  这是JV算法的第三步，使用匈牙利算法的核心思想进行增广
 *  \param n 问题规模
 *  \param cost 代价矩阵
 *  \param n_free_rows 未分配行数
 *  \param free_rows 未分配行数组
 *  \param x 行分配数组
 *  \param y 列分配数组
 *  \param v 列最小值数组
 *  \return 错误代码，0表示成功
 */
int_t _ca_dense(
	const uint_t n, cost_t *cost[],
	const uint_t n_free_rows,
	int_t *free_rows, int_t *x, int_t *y, cost_t *v)
{
	int_t *pred;

	NEW(pred, int_t, n);

	// 对每个未分配的行寻找增广路径
	for (int_t *pfree_i = free_rows; pfree_i < free_rows + n_free_rows; pfree_i++) {
		int_t i = -1, j;
		uint_t k = 0;

		PRINTF("looking at free_i=%d\n", *pfree_i);
		// 寻找从该行开始的增广路径
		j = find_path_dense(n, cost, *pfree_i, y, v, pred);
		ASSERT(j >= 0);
		ASSERT(j < n);
		
		// 沿增广路径进行增广
		while (i != *pfree_i) {
			PRINTF("augment %d\n", j);
			PRINT_INDEX_ARRAY(pred, n);
			i = pred[j];              // 获取前驱行
			PRINTF("y[%d]=%d -> %d\n", j, y[j], i);
			y[j] = i;                 // 更新列分配
			PRINT_INDEX_ARRAY(x, n);
			SWAP_INDICES(j, x[i]);    // 交换索引
			k++;
			if (k >= n) {
				ASSERT(FALSE);
			}
		}
	}
	FREE(pred);
	return 0;
}


/** 求解稠密LAP问题的主函数.
 *  实现了Jonker-Volgenant算法求解线性分配问题
 *  \param n 问题规模(矩阵大小)
 *  \param cost 代价矩阵
 *  \param x 行分配结果数组，x[i]表示分配给行i的列索引
 *  \param y 列分配结果数组，y[j]表示分配给列j的行索引
 *  \return 错误代码，0表示成功
 */
int lapjv_internal(
	const uint_t n, cost_t *cost[],
	int_t *x, int_t *y)
{
	int ret;
	int_t *free_rows;
	cost_t *v;

	NEW(free_rows, int_t, n);
	NEW(v, cost_t, n);
	
	// 第一步：列简化和简化转移
	ret = _ccrrt_dense(n, cost, free_rows, x, y, v);
	int i = 0;
	
	// 第二步：增广行简化（最多执行2次）
	while (ret > 0 && i < 2) {
		ret = _carr_dense(n, cost, ret, free_rows, x, y, v);
		i++;
	}
	
	// 第三步：如果还有未分配的行，则进行增广
	if (ret > 0) {
		ret = _ca_dense(n, cost, ret, free_rows, x, y, v);
	}
	
	FREE(v);
	FREE(free_rows);
	return ret;
}