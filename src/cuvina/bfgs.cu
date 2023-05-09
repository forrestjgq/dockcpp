
#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"
namespace dock {
__device__ void minus_mat_vec_product(const		matrix* h,
	const		change_cl* in,
	change_cl* out
) {
	int n = h->dim;
	for (int i = 0; i < n; i++) {
		float sum = 0;
		for (int j = 0; j < n; j++) {
			sum += h->data[index_permissive(h, i, j)] * find_change_index_read(in, j);
		}
		find_change_index_write(out, i, -sum);
	}
}


__device__ inline float scalar_product(const	change_cl* a,
	const	change_cl* b,
	int			n
) {
	float tmp = 0;
	for (int i = 0; i < n; i++) {
		tmp += find_change_index_read(a, i) * find_change_index_read(b, i);
	}
	return tmp;
}


__device__ float line_search(m_cl* m_cl_gpu,
	p_cl* p_cl_gpu,
	ig_cl* ig_cl_gpu,
	int				n,
	const	output_type_cl* x,
	const	change_cl* g,
	const	float			f0,
	const	change_cl* p,
	output_type_cl* x_new,
	change_cl* g_new,
	float* f1,
	const	float			epsilon_fl,
	const	float* hunt_cap
) {
	const float c0 = 0.0001;
	const int max_trials = 10;
	const float multiplier = 0.5;
	float alpha = 1;

	const float pg = scalar_product(p, g, n);

	for (int trial = 0; trial < max_trials; trial++) {

		output_type_cl_init_with_output(x_new, x);

		output_type_cl_increment(x_new, p, alpha, epsilon_fl);

		*f1 = m_eval_deriv(x_new,
			g_new,
			m_cl_gpu,
			p_cl_gpu,
			ig_cl_gpu,
			hunt_cap,
			epsilon_fl
		);

		if (*f1 - f0 < c0 * alpha * pg)
			break;
		alpha *= multiplier;
	}
	return alpha;
}


__device__  bool bfgs_update(matrix* h,
	const	change_cl* p,
	const	change_cl* y,
	const	float		alpha,
	const	float		epsilon_fl
) {

	const float yp = scalar_product(y, p, h->dim);

	if (alpha * yp < epsilon_fl) return false;
	change_cl minus_hy;
	change_cl_init_with_change(&minus_hy, y);
	minus_mat_vec_product(h, y, &minus_hy);
	const float yhy = -scalar_product(y, &minus_hy, h->dim);
	const float r = 1 / (alpha * yp);
	const int n = 6 + p->lig_torsion_size;

	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			float tmp = alpha * r * (find_change_index_read(&minus_hy, i) * find_change_index_read(p, j)
				+ find_change_index_read(&minus_hy, j) * find_change_index_read(p, i)) +
				+alpha * alpha * (r * r * yhy + r) * find_change_index_read(p, i) * find_change_index_read(p, j);

			h->data[i + j * (j + 1) / 2] += tmp;
		}
	}

	return true;
}



__device__  void bfgs(output_type_cl* x,
	change_cl* g,
	m_cl* m_cl_gpu,
	p_cl* p_cl_gpu,
	ig_cl* ig_cl_gpu,
	const	float* hunt_cap,
	const	float			epsilon_fl,
	const	int				max_steps
)
{
	int n = 3 + 3 + x->lig_torsion_size; // the dimensions of matirx

	matrix h;
	matrix_init(&h, n, 0);
	matrix_set_diagonal(&h, 1);

	change_cl g_new;
	change_cl_init_with_change(&g_new, g);

	output_type_cl x_new;
	output_type_cl_init_with_output(&x_new, x);

	float f0 = m_eval_deriv(x,
		g,
		m_cl_gpu,
		p_cl_gpu,
		ig_cl_gpu,
		hunt_cap,
		epsilon_fl
	);

	float f_orig = f0;
	// Init g_orig, x_orig
	change_cl g_orig;
	change_cl_init_with_change(&g_orig, g);
	output_type_cl x_orig;
	output_type_cl_init_with_output(&x_orig, x);
	// Init p
	change_cl p;
	change_cl_init_with_change(&p, g);

	float f_values[MAX_NUM_OF_BFGS_STEPS + 1];
	f_values[0] = f0;

	for (int step = 0; step < max_steps; step++) {

		minus_mat_vec_product(&h, g, &p);
		float f1 = 0;

		const float alpha = line_search(m_cl_gpu,
			p_cl_gpu,
			ig_cl_gpu,
			n,
			x,
			g,
			f0,
			&p,
			&x_new,
			&g_new,
			&f1,
			epsilon_fl,
			hunt_cap
		);

		change_cl y;
		change_cl_init_with_change(&y, &g_new);
		// subtract_change
		for (int i = 0; i < n; i++) {
			float tmp = find_change_index_read(&y, i) - find_change_index_read(g, i);
			find_change_index_write(&y, i, tmp);
		}
		f_values[step + 1] = f1;
		f0 = f1;
		output_type_cl_init_with_output(x, &x_new);
		if (!(sqrt(scalar_product(g, g, n)) >= 1e-5))break;
		change_cl_init_with_change(g, &g_new);

		if (step == 0) {
			float yy = scalar_product(&y, &y, n);
			if (fabs(yy) > epsilon_fl) {
				matrix_set_diagonal(&h, alpha * scalar_product(&y, &p, n) / yy);
			}
		}

		bool h_updated = bfgs_update(&h, &p, &y, alpha, epsilon_fl);
	}

	if (!(f0 <= f_orig)) {
		f0 = f_orig;
		output_type_cl_init_with_output(x, &x_orig);
		change_cl_init_with_change(g, &g_orig);
	}

	// write output_type_cl enerdasgy
	x->e = f0;
}
};