/*

   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Dr. Oleg Trott <ot14@columbia.edu>, 
           The Olson Lab, 
           The Scripps Research Institute

*/

#ifndef VINA_BFGS_H
#define VINA_BFGS_H

#include "matrix.h"
#define BFGSDEBUG 0
typedef triangular_matrix<fl> flmat;

template<typename Change>
void minus_mat_vec_product(const flmat& m, const Change& in, Change& out) {
	sz n = m.dim();
	VINA_FOR(i, n) {
		fl sum = 0;
		VINA_FOR(j, n) {
			// MCDBG("mmvps %ld %ld: h[%ld] %f * %f", i, j, m.index_permissive(i, j), m(m.index_permissive(i, j)), in(j));
			sum += m(m.index_permissive(i, j)) * in(j);
		}
		DBG("minus_mat_vec_product sum %ld: %f", i, sum);
		out(i) = -sum;
	}
}

template<typename Change>
inline fl scalar_product(const Change& a, const Change& b, sz n) {
	fl tmp = 0;
	VINA_FOR(i, n)
		tmp += a(i) * b(i);
	return tmp;
}
template<typename Change>
inline fl scalar_product1(const Change& a, const Change& b, sz n) {
	fl tmp = 0;
	VINA_FOR(i, n) {
		auto tmp1 = a(i) * b(i);
		tmp+= tmp1;
		// printf("scalar %lu %f * %f = %f, sum %f\n", i, a(i), b(i), tmp1, tmp);
	}
	return tmp;
}

template<typename Change>
inline bool bfgs_update(flmat& h, const Change& p, const Change& y, const fl alpha) {
	const fl yp  = scalar_product(y, p, h.dim());
	if(alpha * yp < epsilon_fl) return false; // FIXME?
	Change minus_hy(y); minus_mat_vec_product(h, y, minus_hy);
	const fl yhy = - scalar_product(y, minus_hy, h.dim());
	const fl r = 1 / (alpha * yp); // 1 / (s^T * y) , where s = alpha * p // FIXME   ... < epsilon
	const sz n = p.num_floats();
	DBG("yp %f yhy %f r %f", yp, yhy, r);
	VINA_FOR(i, n)
		VINA_RANGE(j, i, n) { // includes i 
			DBG("bfgs before update [%ld,%ld]:h: %f", i, j, h(i, j));
			h(i, j) +=   alpha * r * (minus_hy(i) * p(j)
	                                + minus_hy(j) * p(i)) +
			           + alpha * alpha * (r*r * yhy  + r) * p(i) * p(j); // s * s == alpha * alpha * p * p
			DBG("bfgs update [%ld,%ld]: p %f %f hy %f %f h: %f", i, j, p(i), p(j), minus_hy(i), minus_hy(j), h(i, j));

		}
	return true;
}

#define MONITOR_TRIAL 0
// #define REQUIRED() (trial == MONITOR_TRIAL)
#define REQUIRED() false
// #define MUSTED() (trial == MONITOR_TRIAL)
#define MUSTED() false
template<typename F, typename Conf, typename Change>
fl line_search(F& f, sz n, const Conf& x, const Change& g, const fl f0, const Change& p, Conf& x_new, Change& g_new, fl& f1, int& evalcount, int step) { // returns alpha
	const fl c0 = 0.0001;
	const unsigned max_trials = 8; // todo
	// const unsigned max_trials = 10;
	const fl multiplier = 0.5;
	fl alpha = 1;

	const fl pg = scalar_product1(p, g, n);
#if BFGSDEBUG
		printf("step %d pg: %f\n", step, pg);
        printf("p:\n");
		p.print();
#endif

	int t = 0;
	VINA_U_FOR(trial, max_trials) {
#if 0//BFGSDEBUG
		printf("step %d trial %d p:\n", step, trial);
		p.print();
		printf("before conf:\n");
		x.print();
#endif
		x_new = x; x_new.increment(p, alpha);
#if 0
		if (REQUIRED()) {
			printf("after conf:\n");
			x_new.print();
		}
#endif
		f1 = f(x_new, g_new);
		evalcount++;
		t++;
#if BFGSDEBUG
		printf("line search step %d trial %d der %f\n", step, trial, f1);
#endif
		if(f1 - f0 < c0 * alpha * pg) {// FIXME check - div by norm(p) ? no? 
			// printf("breaks at trial %u\n", trial);
			break;
		}
		alpha *= multiplier;
	}
	DBG("cpu line search tries %d times, alpha %f f1 %f", t, alpha, f1);
	return alpha;
}

inline void set_diagonal(flmat& m, fl x) {
	VINA_FOR(i, m.dim())
		m(i, i) = x;
}

template<typename Change>
void subtract_change(Change& b, const Change& a, sz n) { // b -= a
	VINA_FOR(i, n)
		b(i) -= a(i);
}

template<typename F, typename Conf, typename Change>
fl bfgs(F& f, Conf& x, Change& g, const unsigned max_steps, const fl average_required_improvement, const sz over,
		int& evalcount) { // x is I/O, final value is returned
	sz n = g.num_floats();
	flmat h(n, 0);
	set_diagonal(h, 1);

	Change g_new(g);
	Conf x_new(x);
	fl f0 = f(x, g);
	evalcount++;
#if BFGSDEBUG
	printf("f0 %f\n", f0);
	printf("c:\n");
	x.print();
	printf("g:\n");
	g.print();
#endif
	fl f_orig = f0;
	Change g_orig(g);
	Conf x_orig(x);

	Change p(g);

	flv f_values; f_values.reserve(max_steps+1);
	f_values.push_back(f0);

	VINA_U_FOR(step, max_steps) {
		minus_mat_vec_product(h, g, p);
		fl f1 = 0;
#if BFGSDEBUG
		printf("CPU step %d\n", step);
		printf("g:\n");
		g.print();
		printf("p:\n");
		p.print();
		printf("c:\n");
		x.print();
#endif
		const fl alpha = line_search(f, n, x, g, f0, p, x_new, g_new, f1, evalcount, step);

#if BFGSDEBUG
		printf("step %u alpha %f f1 %f\n", step, alpha, f1);
		f.print();
		printf("alpha %f f1 %f\n", alpha, f1);
		printf("c_new:\n");
		x_new.print();
		printf("g:\n");
		g.print();
		printf("g_new:\n");
		g_new.print();
		printf("\n\n\n\n");
#endif
		Change y(g_new); subtract_change(y, g, n);
#if 0
		MCDBG("y in step %d", step);
		y.print();
#endif

		f_values.push_back(f1);
		f0 = f1;
		x = x_new;
		// auto sg = std::sqrt(scalar_product(g, g, n));
		// DBG("step %d sqrt g %f", step,sg);
		if(!(std::sqrt(scalar_product(g, g, n)) >= 1e-5))  {
			// printf("cpu breaks at step %u\n", step);
			break; // breaks for nans too // FIXME !!?? 
		}
		g = g_new; // ?
#if 0
		MCDBG("g new in step %d", step);
		g.print();
#endif

		if(step == 0) {
			const fl yy = scalar_product(y, y, n);
			DBG("yy %f", yy);
			if(std::abs(yy) > epsilon_fl) {
				DBG("step %d set diag %f", step, alpha * scalar_product(y, p, n) / yy);
				set_diagonal(h, alpha * scalar_product(y, p, n) / yy);
			}
		}

		// bool h_updated =
		 bfgs_update(h, p, y, alpha);
	}
	if(!(f0 <= f_orig)) { // succeeds for nans too
		// printf("restore f0\n");
		f0 = f_orig;
		x = x_orig;
		g = g_orig;
	}
	return f0;
}

#endif
