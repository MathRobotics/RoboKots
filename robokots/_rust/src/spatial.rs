pub(crate) fn eye3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

pub(crate) fn set_eye3(flat: &mut [f64], index: usize) {
    set_mat3(flat, index, eye3());
}

pub(crate) fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n == 0.0 {
        [1.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

pub(crate) fn quat_to_rot(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let w = q[0] / n;
    let x = q[1] / n;
    let y = q[2] / n;
    let z = q[3] / n;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

pub(crate) fn eye4() -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub(crate) fn mat4_from_rot_pos(r: [[f64; 3]; 3], p: [f64; 3]) -> [[f64; 4]; 4] {
    [
        [r[0][0], r[0][1], r[0][2], p[0]],
        [r[1][0], r[1][1], r[1][2], p[1]],
        [r[2][0], r[2][1], r[2][2], p[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub(crate) fn mat3_from_mat4(a: [[f64; 4]; 4]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [a[2][0], a[2][1], a[2][2]],
    ]
}

pub(crate) fn mat4_mul(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] =
                a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c] + a[r][3] * b[3][c];
        }
    }
    out
}

pub(crate) fn mat4_mul_cmtm_block(
    a: [[f64; 4]; 4],
    b: [[f64; 4]; 4],
    a_has_base_row: bool,
) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for r in 0..3 {
        out[r][0] = a[r][0] * b[0][0] + a[r][1] * b[1][0] + a[r][2] * b[2][0];
        out[r][1] = a[r][0] * b[0][1] + a[r][1] * b[1][1] + a[r][2] * b[2][1];
        out[r][2] = a[r][0] * b[0][2] + a[r][1] * b[1][2] + a[r][2] * b[2][2];
        out[r][3] = a[r][0] * b[0][3] + a[r][1] * b[1][3] + a[r][2] * b[2][3] + a[r][3] * b[3][3];
    }
    if a_has_base_row {
        out[3] = b[3];
    }
    out
}

pub(crate) fn mat4_mul_hat_se3(a: [[f64; 4]; 4], v: [f64; 6]) -> [[f64; 4]; 4] {
    let wx = v[0];
    let wy = v[1];
    let wz = v[2];
    let vx = v[3];
    let vy = v[4];
    let vz = v[5];
    let mut out = [[0.0; 4]; 4];
    for r in 0..3 {
        out[r][0] = a[r][1] * wz - a[r][2] * wy;
        out[r][1] = -a[r][0] * wz + a[r][2] * wx;
        out[r][2] = a[r][0] * wy - a[r][1] * wx;
        out[r][3] = a[r][0] * vx + a[r][1] * vy + a[r][2] * vz;
    }
    out
}

pub(crate) fn mat4_inv_se3(a: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let r = [
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [a[2][0], a[2][1], a[2][2]],
    ];
    let rt = mat3_transpose(r);
    let p = [a[0][3], a[1][3], a[2][3]];
    mat4_from_rot_pos(rt, scale3(mat3_vec(rt, p), -1.0))
}

pub(crate) fn set_mat4(flat: &mut [f64], index: usize, mat: [[f64; 4]; 4]) {
    let start = index * 16;
    for r in 0..4 {
        for c in 0..4 {
            flat[start + r * 4 + c] = mat[r][c];
        }
    }
}

pub(crate) fn mat4_from_flat(flat: &[f64], index: usize) -> [[f64; 4]; 4] {
    let start = index * 16;
    [
        [
            flat[start],
            flat[start + 1],
            flat[start + 2],
            flat[start + 3],
        ],
        [
            flat[start + 4],
            flat[start + 5],
            flat[start + 6],
            flat[start + 7],
        ],
        [
            flat[start + 8],
            flat[start + 9],
            flat[start + 10],
            flat[start + 11],
        ],
        [
            flat[start + 12],
            flat[start + 13],
            flat[start + 14],
            flat[start + 15],
        ],
    ]
}

pub(crate) fn hat_se3(v: [f64; 6]) -> [[f64; 4]; 4] {
    [
        [0.0, -v[2], v[1], v[3]],
        [v[2], 0.0, -v[0], v[4]],
        [-v[1], v[0], 0.0, v[5]],
        [0.0, 0.0, 0.0, 0.0],
    ]
}

pub(crate) fn vee_se3(m: [[f64; 4]; 4]) -> [f64; 6] {
    [
        0.5 * (m[2][1] - m[1][2]),
        0.5 * (m[0][2] - m[2][0]),
        0.5 * (m[1][0] - m[0][1]),
        m[0][3],
        m[1][3],
        m[2][3],
    ]
}

pub(crate) fn cmtm_vecs_slice(flat: &[f64], index: usize, order: usize) -> &[f64] {
    let len = (order - 1) * 6;
    let start = index * len;
    &flat[start..start + len]
}

pub(crate) fn set_cmtm_vecs_flat(flat: &mut [f64], index: usize, order: usize, vecs: &[f64]) {
    let len = (order - 1) * 6;
    let start = index * len;
    flat[start..start + len].copy_from_slice(vecs);
}

pub(crate) fn cmtm_multiply_into(
    l_mat: [[f64; 4]; 4],
    l_vecs: &[f64],
    r_mat: [[f64; 4]; 4],
    r_vecs: &[f64],
    order: usize,
    fact: &[f64],
    l_blocks: &mut [[[f64; 4]; 4]],
    r_blocks: &mut [[[f64; 4]; 4]],
    out_blocks: &mut [[[f64; 4]; 4]],
    hats: &mut [[[f64; 4]; 4]],
    out_vecs: &mut [f64],
) -> [[f64; 4]; 4] {
    cmtm_mat_blocks_into(l_mat, l_vecs, order, fact, l_blocks);
    cmtm_mat_blocks_into(r_mat, r_vecs, order, fact, r_blocks);
    lower_toeplitz_product_mat4_cmtm_into(l_blocks, r_blocks, order, out_blocks);
    cmtm_from_mat_blocks_into(out_blocks, order, fact, hats, out_vecs)
}

/// Linearise a CMTM product for one tangent direction.
///
/// This is an analytic product-rule implementation: `d_*` values are exact
/// differentials, never perturbed primal evaluations.  The caller owns the
/// primal scratch buffers because the following CMTM dynamics pass reuses
/// them; tangent block buffers are intentionally passed separately so several
/// RHS columns can share the primal calculation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmtm_multiply_tangent_into(
    l_mat: [[f64; 4]; 4],
    l_vecs: &[f64],
    dl_mat: [[f64; 4]; 4],
    dl_vecs: &[f64],
    r_mat: [[f64; 4]; 4],
    r_vecs: &[f64],
    dr_mat: [[f64; 4]; 4],
    dr_vecs: &[f64],
    order: usize,
    fact: &[f64],
    l_blocks: &mut [[[f64; 4]; 4]],
    r_blocks: &mut [[[f64; 4]; 4]],
    out_blocks: &mut [[[f64; 4]; 4]],
    hats: &mut [[[f64; 4]; 4]],
    out_vecs: &mut [f64],
    dl_blocks: &mut [[[f64; 4]; 4]],
    dr_blocks: &mut [[[f64; 4]; 4]],
    dout_blocks: &mut [[[f64; 4]; 4]],
    dout_vecs: &mut [f64],
) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
    let out_mat = cmtm_multiply_into(
        l_mat, l_vecs, r_mat, r_vecs, order, fact, l_blocks, r_blocks,
        out_blocks, hats, out_vecs,
    );
    cmtm_mat_blocks_tangent_into(l_blocks, l_vecs, dl_mat, dl_vecs, order, fact, dl_blocks);
    cmtm_mat_blocks_tangent_into(r_blocks, r_vecs, dr_mat, dr_vecs, order, fact, dr_blocks);
    for k in 0..order {
        let mut acc = [[0.0; 4]; 4];
        for i in 0..=k {
            acc = add_mat4(acc, mat4_mul(dl_blocks[i], r_blocks[k - i]));
            acc = add_mat4(acc, mat4_mul(l_blocks[i], dr_blocks[k - i]));
        }
        dout_blocks[k] = acc;
    }
    let dout_mat = cmtm_from_mat_blocks_tangent_into(
        out_blocks,
        dout_blocks,
        order,
        fact,
        hats,
        out_vecs,
        dout_vecs,
    );
    (out_mat, dout_mat)
}

/// Tangent recurrence matching `cmtm_mat_blocks_into`.
pub(crate) fn cmtm_mat_blocks_tangent_into(
    blocks: &[[[f64; 4]; 4]],
    vecs: &[f64],
    dmat: [[f64; 4]; 4],
    dvecs: &[f64],
    order: usize,
    fact: &[f64],
    dblocks: &mut [[[f64; 4]; 4]],
) {
    if order == 0 {
        return;
    }
    dblocks[0] = dmat;
    for k in 1..order {
        let mut acc = [[0.0; 4]; 4];
        for i in 0..k {
            let vec = scale6(vec6_from_flat(vecs, i), 1.0 / fact[i]);
            let dvec = scale6(vec6_from_flat(dvecs, i), 1.0 / fact[i]);
            acc = add_mat4(acc, mat4_mul_hat_se3(dblocks[k - i - 1], vec));
            acc = add_mat4(acc, mat4_mul_hat_se3(blocks[k - i - 1], dvec));
        }
        dblocks[k] = scale_mat4(acc, 1.0 / k as f64);
    }
}

/// Differentiate the block-to-CMTM-vector recovery recurrence.
pub(crate) fn cmtm_from_mat_blocks_tangent_into(
    blocks: &[[[f64; 4]; 4]],
    dblocks: &[[[f64; 4]; 4]],
    order: usize,
    fact: &[f64],
    hats: &mut [[[f64; 4]; 4]],
    out_vecs: &[f64],
    dout_vecs: &mut [f64],
) -> [[f64; 4]; 4] {
    if order == 0 {
        return [[0.0; 4]; 4];
    }
    let elem_inv = mat4_inv_se3(blocks[0]);
    let delem_inv = scale_mat4(mat4_mul(mat4_mul(elem_inv, dblocks[0]), elem_inv), -1.0);
    dout_vecs[..(order - 1) * 6].fill(0.0);
    for i in 0..order - 1 {
        let mut m_tmp = [[0.0; 4]; 4];
        let mut dm_tmp = [[0.0; 4]; 4];
        for j in 0..i {
            let vec = scale6(vec6_from_flat(out_vecs, j), 1.0 / fact[j]);
            let dvec = scale6(vec6_from_flat(dout_vecs, j), 1.0 / fact[j]);
            m_tmp = add_mat4(m_tmp, mat4_mul_hat_se3(blocks[i - j], vec));
            dm_tmp = add_mat4(dm_tmp, mat4_mul_hat_se3(dblocks[i - j], vec));
            dm_tmp = add_mat4(dm_tmp, mat4_mul_hat_se3(blocks[i - j], dvec));
        }
        let inner = sub_mat4(scale_mat4(blocks[i + 1], (i + 1) as f64), m_tmp);
        let dinner = sub_mat4(scale_mat4(dblocks[i + 1], (i + 1) as f64), dm_tmp);
        let delta = mat4_mul_cmtm_block(elem_inv, inner, true);
        let ddelta = add_mat4(mat4_mul(delem_inv, inner), mat4_mul(elem_inv, dinner));
        hats[i] = delta;
        set_vec6_flat(dout_vecs, i, scale6(vee_se3(ddelta), fact[i]));
    }
    dblocks[0]
}

#[allow(dead_code)]
pub(crate) fn cmtm_relative_into(
    l_mat: [[f64; 4]; 4],
    l_vecs: &[f64],
    r_mat: [[f64; 4]; 4],
    r_vecs: &[f64],
    order: usize,
    fact: &[f64],
    l_inv_blocks: &mut [[[f64; 4]; 4]],
    r_blocks: &mut [[[f64; 4]; 4]],
    out_blocks: &mut [[[f64; 4]; 4]],
    hats: &mut [[[f64; 4]; 4]],
    out_vecs: &mut [f64],
) -> [[f64; 4]; 4] {
    cmtm_mat_inv_blocks_into(l_mat, l_vecs, order, fact, l_inv_blocks);
    cmtm_mat_blocks_into(r_mat, r_vecs, order, fact, r_blocks);
    lower_toeplitz_product_mat4_cmtm_into(l_inv_blocks, r_blocks, order, out_blocks);
    cmtm_from_mat_blocks_into(out_blocks, order, fact, hats, out_vecs)
}

pub(crate) fn lower_toeplitz_product_mat4_cmtm_into(
    left: &[[[f64; 4]; 4]],
    right: &[[[f64; 4]; 4]],
    order: usize,
    out: &mut [[[f64; 4]; 4]],
) {
    for k in 0..order {
        let mut acc = [[0.0; 4]; 4];
        for i in 0..=k {
            acc = add_mat4(acc, mat4_mul_cmtm_block(left[i], right[k - i], i == 0));
        }
        out[k] = acc;
    }
}

pub(crate) fn cmtm_mat_blocks_into(
    elem_mat: [[f64; 4]; 4],
    vecs: &[f64],
    order: usize,
    fact: &[f64],
    blocks: &mut [[[f64; 4]; 4]],
) {
    if order == 0 {
        return;
    }
    blocks[0] = elem_mat;
    for k in 1..order {
        let mut acc = [[0.0; 4]; 4];
        for i in 0..k {
            let cm_vec = scale6(vec6_from_flat(vecs, i), 1.0 / fact[i]);
            acc = add_mat4(acc, mat4_mul_hat_se3(blocks[k - i - 1], cm_vec));
        }
        blocks[k] = scale_mat4(acc, 1.0 / k as f64);
    }
}

#[allow(dead_code)]
pub(crate) fn cmtm_mat_inv_blocks_into(
    elem_mat: [[f64; 4]; 4],
    vecs: &[f64],
    order: usize,
    fact: &[f64],
    blocks: &mut [[[f64; 4]; 4]],
) {
    if order == 0 {
        return;
    }
    blocks[0] = mat4_inv_se3(elem_mat);
    for k in 1..order {
        let mut acc = [[0.0; 4]; 4];
        for i in 0..k {
            let cm_vec = scale6(vec6_from_flat(vecs, i), 1.0 / fact[i]);
            acc = sub_mat4(acc, mat4_mul(hat_se3(cm_vec), blocks[k - i - 1]));
        }
        blocks[k] = scale_mat4(acc, 1.0 / k as f64);
    }
}

pub(crate) fn cmtm_mat_adj_wrench_blocks_into(
    elem_mat: [[f64; 4]; 4],
    vecs: &[f64],
    order: usize,
    fact: &[f64],
    blocks: &mut [[[f64; 6]; 6]],
) {
    if order == 0 {
        return;
    }
    blocks[0] = mat_adj_wrench_from_mat4(elem_mat);
    for k in 1..order {
        let mut acc = [[0.0; 6]; 6];
        for i in 0..k {
            let cm_vec = scale6(vec6_from_flat(vecs, i), 1.0 / fact[i]);
            acc = add_mat6(acc, mat6_mul(blocks[k - i - 1], hat_adj_wrench(cm_vec)));
        }
        blocks[k] = scale_mat6(acc, 1.0 / k as f64);
    }
}

/// Apply a CMTM wrench transform and its analytic directional derivative.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmtm_apply_mat_adj_wrench_tangent_into(
    mat: [[f64; 4]; 4], vecs: &[f64], dmat: [[f64; 4]; 4], dvecs: &[f64],
    rhs: &[f64], drhs: &[f64], order: usize, fact: &[f64],
    blocks: &mut [[[f64; 6]; 6]], dblocks: &mut [[[f64; 6]; 6]], out: &mut [f64], dout: &mut [f64],
) {
    cmtm_mat_adj_wrench_blocks_into(mat, vecs, order, fact, blocks);
    dblocks[0] = mat_adj_wrench_tangent_from_mat4(mat, dmat);
    for k in 1..order {
        let mut acc = [[0.0; 6]; 6];
        for i in 0..k {
            let v = scale6(vec6_from_flat(vecs, i), 1.0 / fact[i]);
            let dv = scale6(vec6_from_flat(dvecs, i), 1.0 / fact[i]);
            acc = add_mat6(acc, mat6_mul(dblocks[k-i-1], hat_adj_wrench(v)));
            acc = add_mat6(acc, mat6_mul(blocks[k-i-1], hat_adj_wrench(dv)));
        }
        dblocks[k] = scale_mat6(acc, 1.0 / k as f64);
    }
    for k in 0..order {
        let mut value = [0.0; 6]; let mut dvalue = [0.0; 6];
        for i in 0..=k {
            // CMTM blocks operate on factorial-scaled coefficients, whereas
            // dynamics stores raw time derivatives.  Match
            // cmtm_accumulate_mat_adj_wrench_series_into exactly.
            let rhs_cm = scale6(vec6_from_flat(rhs, k - i), 1.0 / fact[k - i]);
            let drhs_cm = scale6(vec6_from_flat(drhs, k - i), 1.0 / fact[k - i]);
            value = add6(value, mat6_vec6(blocks[i], rhs_cm));
            dvalue = add6(dvalue, mat6_vec6(dblocks[i], rhs_cm));
            dvalue = add6(dvalue, mat6_vec6(blocks[i], drhs_cm));
        }
        set_vec6_flat(out, k, scale6(value, fact[k]));
        set_vec6_flat(dout, k, scale6(dvalue, fact[k]));
    }
}

fn mat_adj_wrench_tangent_from_mat4(mat: [[f64; 4]; 4], dmat: [[f64; 4]; 4]) -> [[f64; 6]; 6] {
    let r = mat3_from_mat4(mat); let dr = mat3_from_mat4(dmat);
    let p = [mat[0][3], mat[1][3], mat[2][3]];
    let dp = [dmat[0][3], dmat[1][3], dmat[2][3]];
    let upper = add_mat3(mat3_mul(skew(dp), r), mat3_mul(skew(p), dr));
    let mut out = [[0.0; 6]; 6];
    for row in 0..3 { for col in 0..3 { out[row][col]=dr[row][col]; out[row][col+3]=upper[row][col]; out[row+3][col+3]=dr[row][col]; } }
    out
}

pub(crate) fn cmtm_mat_inv_adj_wrench_blocks_into(
    elem_mat: [[f64; 4]; 4],
    vecs: &[f64],
    order: usize,
    fact: &[f64],
    blocks: &mut [[[f64; 6]; 6]],
) {
    if order == 0 {
        return;
    }
    blocks[0] = mat_inv_adj_wrench_from_mat4(elem_mat);
    for k in 1..order {
        let mut acc = [[0.0; 6]; 6];
        for i in 0..k {
            let cm_vec = scale6(vec6_from_flat(vecs, i), 1.0 / fact[i]);
            acc = sub_mat6(acc, mat6_mul(hat_adj_wrench(cm_vec), blocks[k - i - 1]));
        }
        blocks[k] = scale_mat6(acc, 1.0 / k as f64);
    }
}

pub(crate) fn cmtm_apply_mat_adj_wrench_with_blocks_into(
    elem_mat: [[f64; 4]; 4],
    vecs: &[f64],
    cm_rhs: &[f64],
    order: usize,
    fact: &[f64],
    blocks: &mut [[[f64; 6]; 6]],
    out: &mut [f64],
) {
    cmtm_mat_adj_wrench_blocks_into(elem_mat, vecs, order, fact, blocks);
    out[..order * 6].fill(0.0);
    for k in 0..order {
        let mut acc = [0.0; 6];
        for i in 0..=k {
            acc = add6(acc, mat6_vec6(blocks[i], vec6_from_flat(cm_rhs, k - i)));
        }
        set_vec6_flat(out, k, acc);
    }
}

#[allow(dead_code)]
pub(crate) fn cmtm_apply_mat_adj_wrench_series_into(
    elem_mat: [[f64; 4]; 4],
    vecs: &[f64],
    cm_rhs: &[f64],
    order: usize,
    fact: &[f64],
    a_blocks: &mut [[[f64; 3]; 3]],
    c_blocks: &mut [[[f64; 3]; 3]],
    out: &mut [f64],
) {
    if order == 0 {
        return;
    }

    let r = [
        [elem_mat[0][0], elem_mat[0][1], elem_mat[0][2]],
        [elem_mat[1][0], elem_mat[1][1], elem_mat[1][2]],
        [elem_mat[2][0], elem_mat[2][1], elem_mat[2][2]],
    ];
    let p = [elem_mat[0][3], elem_mat[1][3], elem_mat[2][3]];
    a_blocks[0] = r;
    c_blocks[0] = mat3_mul(skew(p), r);

    for k in 1..order {
        let mut acc_a = [[0.0; 3]; 3];
        let mut acc_c = [[0.0; 3]; 3];
        for i in 0..k {
            let cm_vec = scale6(vec6_from_flat(vecs, i), 1.0 / fact[i]);
            let w_hat = skew([cm_vec[0], cm_vec[1], cm_vec[2]]);
            let v_hat = skew([cm_vec[3], cm_vec[4], cm_vec[5]]);
            let prev = k - i - 1;
            acc_a = add_mat3(acc_a, mat3_mul(a_blocks[prev], w_hat));
            acc_c = add_mat3(
                acc_c,
                add_mat3(
                    mat3_mul(a_blocks[prev], v_hat),
                    mat3_mul(c_blocks[prev], w_hat),
                ),
            );
        }
        let scale = 1.0 / k as f64;
        a_blocks[k] = scale_mat3(acc_a, scale);
        c_blocks[k] = scale_mat3(acc_c, scale);
    }

    out[..order * 6].fill(0.0);
    for k in 0..order {
        let mut acc = [0.0; 6];
        for i in 0..=k {
            let rhs = vec6_from_flat(cm_rhs, k - i);
            let torque = [rhs[0], rhs[1], rhs[2]];
            let force = [rhs[3], rhs[4], rhs[5]];
            let out_torque = add3(mat3_vec(a_blocks[i], torque), mat3_vec(c_blocks[i], force));
            let out_force = mat3_vec(a_blocks[i], force);
            acc = add6(
                acc,
                [
                    out_torque[0],
                    out_torque[1],
                    out_torque[2],
                    out_force[0],
                    out_force[1],
                    out_force[2],
                ],
            );
        }
        set_vec6_flat(out, k, acc);
    }
}

pub(crate) fn cmtm_accumulate_mat_adj_wrench_series_into(
    elem_mat: [[f64; 4]; 4],
    vecs: &[f64],
    raw_rhs: &[f64],
    order: usize,
    fact: &[f64],
    scaled_vecs: &mut [f64],
    a_blocks: &mut [[[f64; 3]; 3]],
    c_blocks: &mut [[[f64; 3]; 3]],
    raw_target: &mut [f64],
) {
    if order == 0 {
        return;
    }

    let r = [
        [elem_mat[0][0], elem_mat[0][1], elem_mat[0][2]],
        [elem_mat[1][0], elem_mat[1][1], elem_mat[1][2]],
        [elem_mat[2][0], elem_mat[2][1], elem_mat[2][2]],
    ];
    let p = [elem_mat[0][3], elem_mat[1][3], elem_mat[2][3]];
    a_blocks[0] = r;
    c_blocks[0] = mat3_mul(skew(p), r);

    for i in 0..order - 1 {
        let scale = 1.0 / fact[i];
        let start = i * 6;
        for j in 0..6 {
            scaled_vecs[start + j] = vecs[start + j] * scale;
        }
    }

    for k in 1..order {
        let mut acc_a = [[0.0; 3]; 3];
        let mut acc_c = [[0.0; 3]; 3];
        for i in 0..k {
            let cm_vec = vec6_from_flat(scaled_vecs, i);
            let prev = k - i - 1;
            let w = [cm_vec[0], cm_vec[1], cm_vec[2]];
            let v = [cm_vec[3], cm_vec[4], cm_vec[5]];
            acc_a = add_mat3(acc_a, mat3_mul_skew_right(a_blocks[prev], w));
            acc_c = add_mat3(
                acc_c,
                add_mat3(
                    mat3_mul_skew_right(a_blocks[prev], v),
                    mat3_mul_skew_right(c_blocks[prev], w),
                ),
            );
        }
        let scale = 1.0 / k as f64;
        a_blocks[k] = scale_mat3(acc_a, scale);
        c_blocks[k] = scale_mat3(acc_c, scale);
    }

    for k in 0..order {
        let mut acc = [0.0; 6];
        for i in 0..=k {
            let rhs = scale6(vec6_from_flat(raw_rhs, k - i), 1.0 / fact[k - i]);
            let torque = [rhs[0], rhs[1], rhs[2]];
            let force = [rhs[3], rhs[4], rhs[5]];
            let out_torque = add3(mat3_vec(a_blocks[i], torque), mat3_vec(c_blocks[i], force));
            let out_force = mat3_vec(a_blocks[i], force);
            acc = add6(
                acc,
                [
                    out_torque[0],
                    out_torque[1],
                    out_torque[2],
                    out_force[0],
                    out_force[1],
                    out_force[2],
                ],
            );
        }
        let start = k * 6;
        let scale = fact[k];
        for i in 0..6 {
            raw_target[start + i] += acc[i] * scale;
        }
    }
}

pub(crate) fn cmtm_wrench_var_jacob_matvec_into(
    elem_mat: [[f64; 4]; 4],
    elem_vecs: &[f64],
    arb_cm_vecs: &[f64],
    rhs: &[f64],
    order: usize,
    inverse: bool,
    transpose: bool,
    fact: &[f64],
    blocks: &mut [[[f64; 6]; 6]],
    tmp: &mut [f64],
    inv_arb: &mut [f64],
    out: &mut [f64],
) {
    if inverse {
        cmtm_mat_inv_adj_wrench_blocks_into(elem_mat, elem_vecs, order, fact, blocks);
        apply_lower_toeplitz_blocks(blocks, arb_cm_vecs, order, inv_arb);
        if transpose {
            apply_hat_cm_commute_wrench_transpose(inv_arb, rhs, order, out);
        } else {
            apply_hat_cm_commute_wrench(inv_arb, rhs, order, out);
        }
        for value in out.iter_mut().take(order * 6) {
            *value = -*value;
        }
        return;
    }

    cmtm_mat_adj_wrench_blocks_into(elem_mat, elem_vecs, order, fact, blocks);
    if transpose {
        apply_lower_toeplitz_blocks_transpose(blocks, rhs, order, tmp);
        apply_hat_cm_commute_wrench_transpose(arb_cm_vecs, tmp, order, out);
    } else {
        apply_hat_cm_commute_wrench(arb_cm_vecs, rhs, order, tmp);
        apply_lower_toeplitz_blocks(blocks, tmp, order, out);
    }
}

pub(crate) fn cmtm_wrench_var_jacob_matmul_rhs_into(
    elem_mat: [[f64; 4]; 4],
    elem_vecs: &[f64],
    arb_cm_vecs: &[f64],
    rhs: &[f64],
    order: usize,
    rhs_dim: usize,
    inverse: bool,
    fact: &[f64],
    blocks: &mut [[[f64; 6]; 6]],
    tmp: &mut [f64],
    inv_arb: &mut [f64],
    rhs_col: &mut [f64],
    out_col: &mut [f64],
    out: &mut [f64],
) {
    if inverse {
        cmtm_mat_inv_adj_wrench_blocks_into(elem_mat, elem_vecs, order, fact, blocks);
        apply_lower_toeplitz_blocks(blocks, arb_cm_vecs, order, inv_arb);
    } else {
        cmtm_mat_adj_wrench_blocks_into(elem_mat, elem_vecs, order, fact, blocks);
    }

    for col in 0..rhs_dim {
        for row in 0..order * 6 {
            rhs_col[row] = rhs[row * rhs_dim + col];
        }
        if inverse {
            apply_hat_cm_commute_wrench(inv_arb, rhs_col, order, out_col);
            for value in out_col.iter_mut().take(order * 6) {
                *value = -*value;
            }
        } else {
            apply_hat_cm_commute_wrench(arb_cm_vecs, rhs_col, order, tmp);
            apply_lower_toeplitz_blocks(blocks, tmp, order, out_col);
        }
        for row in 0..order * 6 {
            out[row * rhs_dim + col] = out_col[row];
        }
    }
}

pub(crate) fn apply_lower_toeplitz_blocks(
    blocks: &[[[f64; 6]; 6]],
    rhs: &[f64],
    order: usize,
    out: &mut [f64],
) {
    out[..order * 6].fill(0.0);
    for k in 0..order {
        let mut acc = [0.0; 6];
        for i in 0..=k {
            acc = add6(acc, mat6_vec6(blocks[i], vec6_from_flat(rhs, k - i)));
        }
        set_vec6_flat(out, k, acc);
    }
}

pub(crate) fn apply_lower_toeplitz_blocks_transpose(
    blocks: &[[[f64; 6]; 6]],
    rhs: &[f64],
    order: usize,
    out: &mut [f64],
) {
    out[..order * 6].fill(0.0);
    for j in 0..order {
        let mut acc = [0.0; 6];
        for i in 0..order - j {
            acc = add6(
                acc,
                mat6_transpose_vec6(blocks[i], vec6_from_flat(rhs, i + j)),
            );
        }
        set_vec6_flat(out, j, acc);
    }
}

pub(crate) fn apply_hat_cm_commute_wrench(
    cm_vecs: &[f64],
    rhs: &[f64],
    order: usize,
    out: &mut [f64],
) {
    out[..order * 6].fill(0.0);
    for k in 0..order {
        let mut acc = [0.0; 6];
        for i in 0..=k {
            acc = add6(
                acc,
                mat6_vec6(
                    hat_commute_adj_wrench(vec6_from_flat(cm_vecs, i)),
                    vec6_from_flat(rhs, k - i),
                ),
            );
        }
        set_vec6_flat(out, k, acc);
    }
}

pub(crate) fn apply_hat_cm_commute_wrench_transpose(
    cm_vecs: &[f64],
    rhs: &[f64],
    order: usize,
    out: &mut [f64],
) {
    out[..order * 6].fill(0.0);
    for j in 0..order {
        let mut acc = [0.0; 6];
        for i in 0..order - j {
            acc = add6(
                acc,
                mat6_transpose_vec6(
                    hat_commute_adj_wrench(vec6_from_flat(cm_vecs, i)),
                    vec6_from_flat(rhs, i + j),
                ),
            );
        }
        set_vec6_flat(out, j, acc);
    }
}

pub(crate) fn cmtm_from_mat_blocks_into(
    blocks: &[[[f64; 4]; 4]],
    order: usize,
    fact: &[f64],
    _hats: &mut [[[f64; 4]; 4]],
    out_vecs: &mut [f64],
) -> [[f64; 4]; 4] {
    let elem_mat = blocks[0];
    let elem_inv = mat4_inv_se3(elem_mat);

    for i in 0..order - 1 {
        let mut m_tmp = [[0.0; 4]; 4];
        for j in 0..i {
            let cm_vec = scale6(vec6_from_flat(out_vecs, j), 1.0 / fact[j]);
            m_tmp = add_mat4(m_tmp, mat4_mul_hat_se3(blocks[i - j], cm_vec));
        }
        let delta = mat4_mul_cmtm_block(
            elem_inv,
            sub_mat4(scale_mat4(blocks[i + 1], (i + 1) as f64), m_tmp),
            true,
        );
        let raw = scale6(vee_se3(delta), fact[i]);
        set_vec6_flat(out_vecs, i, raw);
    }

    elem_mat
}

pub(crate) fn fill_factorial_table(fact: &mut [f64]) {
    if fact.is_empty() {
        return;
    }
    fact[0] = 1.0;
    for i in 1..fact.len() {
        fact[i] = fact[i - 1] * i as f64;
    }
}

pub(crate) fn momentum_from_velocity_into(
    inertia: [[f64; 6]; 6],
    vel: &[f64],
    order: usize,
    out: &mut [f64],
) {
    out[..order * 6].fill(0.0);
    for k in 0..order {
        set_vec6_flat(out, k, mat6_vec6(inertia, vec6_from_flat(vel, k)));
    }
}

pub(crate) fn force_from_velocity_momentum_into(
    vel: &[f64],
    momentum: &[f64],
    force_order: usize,
    fact: &[f64],
    out: &mut [f64],
) {
    out[..force_order * 6].fill(0.0);
    for k in 0..force_order {
        let mut conv_cm = [0.0; 6];
        for i in 0..=k {
            let vel_cm = scale6(vec6_from_flat(vel, i), 1.0 / fact[i]);
            let mom_cm = scale6(vec6_from_flat(momentum, k - i), 1.0 / fact[k - i]);
            conv_cm = add6(conv_cm, hat_adj_wrench_vec6(vel_cm, mom_cm));
        }
        let force = add6(vec6_from_flat(momentum, k + 1), scale6(conv_cm, fact[k]));
        set_vec6_flat(out, k, force);
    }
}

/// Analytic directional derivative of `force_from_velocity_momentum_into`.
pub(crate) fn force_from_velocity_momentum_tangent_into(
    vel: &[f64],
    dvel: &[f64],
    momentum: &[f64],
    dmomentum: &[f64],
    force_order: usize,
    fact: &[f64],
    out: &mut [f64],
) {
    out[..force_order * 6].fill(0.0);
    for k in 0..force_order {
        let mut conv = [0.0; 6];
        for i in 0..=k {
            let vel_cm = scale6(vec6_from_flat(vel, i), 1.0 / fact[i]);
            let dvel_cm = scale6(vec6_from_flat(dvel, i), 1.0 / fact[i]);
            let mom_cm = scale6(vec6_from_flat(momentum, k - i), 1.0 / fact[k - i]);
            let dmom_cm = scale6(
                vec6_from_flat(dmomentum, k - i),
                1.0 / fact[k - i],
            );
            conv = add6(conv, hat_adj_wrench_vec6(dvel_cm, mom_cm));
            conv = add6(conv, hat_adj_wrench_vec6(vel_cm, dmom_cm));
        }
        set_vec6_flat(
            out,
            k,
            add6(
                vec6_from_flat(dmomentum, k + 1),
                scale6(conv, fact[k]),
            ),
        );
    }
}

#[allow(dead_code)]
pub(crate) fn cm_scale_vecs_into(vecs: &[f64], order: usize, fact: &[f64], out: &mut [f64]) {
    out[..order * 6].fill(0.0);
    for (k, scale) in fact.iter().enumerate().take(order) {
        set_vec6_flat(out, k, scale6(vec6_from_flat(vecs, k), 1.0 / scale));
    }
}

#[allow(dead_code)]
pub(crate) fn add_factorial_scaled_cmvec_with_fact(
    raw_target: &mut [f64],
    cm_vecs: &[f64],
    order: usize,
    fact: &[f64],
) {
    for (k, scale) in fact.iter().enumerate().take(order) {
        let start = k * 6;
        for i in 0..6 {
            raw_target[start + i] += cm_vecs[start + i] * scale;
        }
    }
}

pub(crate) fn cmvec_slice(flat: &[f64], index: usize, order: usize) -> &[f64] {
    let len = order * 6;
    let start = index * len;
    &flat[start..start + len]
}

pub(crate) fn set_cmvec_flat(flat: &mut [f64], index: usize, order: usize, vecs: &[f64]) {
    let len = order * 6;
    let start = index * len;
    flat[start..start + len].copy_from_slice(vecs);
}

pub(crate) fn vec6_from_flat(flat: &[f64], index: usize) -> [f64; 6] {
    let start = index * 6;
    [
        flat[start],
        flat[start + 1],
        flat[start + 2],
        flat[start + 3],
        flat[start + 4],
        flat[start + 5],
    ]
}

pub(crate) fn set_vec6_flat(flat: &mut [f64], index: usize, value: [f64; 6]) {
    let start = index * 6;
    flat[start] = value[0];
    flat[start + 1] = value[1];
    flat[start + 2] = value[2];
    flat[start + 3] = value[3];
    flat[start + 4] = value[4];
    flat[start + 5] = value[5];
}

pub(crate) fn add6(a: [f64; 6], b: [f64; 6]) -> [f64; 6] {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
        a[5] + b[5],
    ]
}

pub(crate) fn scale6(a: [f64; 6], s: f64) -> [f64; 6] {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s, a[4] * s, a[5] * s]
}

pub(crate) fn mat_adj_wrench_from_mat4(a: [[f64; 4]; 4]) -> [[f64; 6]; 6] {
    let r = [
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [a[2][0], a[2][1], a[2][2]],
    ];
    let p = [a[0][3], a[1][3], a[2][3]];
    let upper = mat3_mul(skew(p), r);
    let mut out = [[0.0; 6]; 6];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = r[row][col];
            out[row][col + 3] = upper[row][col];
            out[row + 3][col + 3] = r[row][col];
        }
    }
    out
}

pub(crate) fn mat_adj_wrench_vec6_from_mat4(a: [[f64; 4]; 4], rhs: [f64; 6]) -> [f64; 6] {
    let r = mat3_from_mat4(a);
    let p = [a[0][3], a[1][3], a[2][3]];
    let torque = [rhs[0], rhs[1], rhs[2]];
    let force = [rhs[3], rhs[4], rhs[5]];
    let rot_torque = mat3_vec(r, torque);
    let rot_force = mat3_vec(r, force);
    let out_torque = add3(rot_torque, cross(p, rot_force));
    [
        out_torque[0],
        out_torque[1],
        out_torque[2],
        rot_force[0],
        rot_force[1],
        rot_force[2],
    ]
}

pub(crate) fn mat_inv_adj_wrench_from_mat4(a: [[f64; 4]; 4]) -> [[f64; 6]; 6] {
    let r = [
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [a[2][0], a[2][1], a[2][2]],
    ];
    let rt = mat3_transpose(r);
    let p = [a[0][3], a[1][3], a[2][3]];
    let upper = scale_mat3(mat3_mul(rt, skew(p)), -1.0);
    let mut out = [[0.0; 6]; 6];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = rt[row][col];
            out[row][col + 3] = upper[row][col];
            out[row + 3][col + 3] = rt[row][col];
        }
    }
    out
}

pub(crate) fn hat_adj_wrench(v: [f64; 6]) -> [[f64; 6]; 6] {
    let w_hat = skew([v[0], v[1], v[2]]);
    let v_hat = skew([v[3], v[4], v[5]]);
    let mut out = [[0.0; 6]; 6];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = w_hat[row][col];
            out[row][col + 3] = v_hat[row][col];
            out[row + 3][col + 3] = w_hat[row][col];
        }
    }
    out
}

pub(crate) fn hat_adj_wrench_vec6(v: [f64; 6], rhs: [f64; 6]) -> [f64; 6] {
    let w = [v[0], v[1], v[2]];
    let lin = [v[3], v[4], v[5]];
    let torque = [rhs[0], rhs[1], rhs[2]];
    let force = [rhs[3], rhs[4], rhs[5]];
    let out_torque = add3(cross(w, torque), cross(lin, force));
    let out_force = cross(w, force);
    [
        out_torque[0],
        out_torque[1],
        out_torque[2],
        out_force[0],
        out_force[1],
        out_force[2],
    ]
}

pub(crate) fn hat_commute_adj_wrench(v: [f64; 6]) -> [[f64; 6]; 6] {
    let w_hat = skew([v[0], v[1], v[2]]);
    let v_hat = skew([v[3], v[4], v[5]]);
    let mut out = [[0.0; 6]; 6];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = -w_hat[row][col];
            out[row][col + 3] = -v_hat[row][col];
            out[row + 3][col] = -v_hat[row][col];
        }
    }
    out
}

pub(crate) fn mat6_vec6(a: [[f64; 6]; 6], v: [f64; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for row in 0..6 {
        out[row] = a[row][0] * v[0]
            + a[row][1] * v[1]
            + a[row][2] * v[2]
            + a[row][3] * v[3]
            + a[row][4] * v[4]
            + a[row][5] * v[5];
    }
    out
}

pub(crate) fn mat6_transpose_vec6(a: [[f64; 6]; 6], v: [f64; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for col in 0..6 {
        out[col] = a[0][col] * v[0]
            + a[1][col] * v[1]
            + a[2][col] * v[2]
            + a[3][col] * v[3]
            + a[4][col] * v[4]
            + a[5][col] * v[5];
    }
    out
}

pub(crate) fn mat6_mul(a: [[f64; 6]; 6], b: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            let mut value = 0.0;
            for k in 0..6 {
                value += a[row][k] * b[k][col];
            }
            out[row][col] = value;
        }
    }
    out
}

pub(crate) fn sub_mat6(a: [[f64; 6]; 6], b: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            out[row][col] = a[row][col] - b[row][col];
        }
    }
    out
}

pub(crate) fn add_mat6(a: [[f64; 6]; 6], b: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            out[row][col] = a[row][col] + b[row][col];
        }
    }
    out
}

pub(crate) fn scale_mat6(a: [[f64; 6]; 6], s: f64) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            out[row][col] = a[row][col] * s;
        }
    }
    out
}

pub(crate) fn add_mat4(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = a[r][c] + b[r][c];
        }
    }
    out
}

pub(crate) fn sub_mat4(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = a[r][c] - b[r][c];
        }
    }
    out
}

pub(crate) fn scale_mat4(a: [[f64; 4]; 4], s: f64) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = a[r][c] * s;
        }
    }
    out
}

pub(crate) fn skew(v: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

pub(crate) fn rot_axis(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let axis = normalize(axis);
    let k = skew(axis);
    let k2 = mat3_mul(k, k);
    let s = angle.sin();
    let c = angle.cos();
    let mut out = eye3();
    for r in 0..3 {
        for col in 0..3 {
            out[r][col] += s * k[r][col] + (1.0 - c) * k2[r][col];
        }
    }
    out
}

pub(crate) fn rot_axis_derivative(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let axis = normalize(axis);
    let k = skew(axis);
    let k2 = mat3_mul(k, k);
    let c = angle.cos();
    let s = angle.sin();
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for col in 0..3 {
            out[r][col] = c * k[r][col] + s * k2[r][col];
        }
    }
    out
}

pub(crate) fn add_mat3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][c] + b[r][c];
        }
    }
    out
}

pub(crate) fn scale_mat3(a: [[f64; 3]; 3], s: f64) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][c] * s;
        }
    }
    out
}

pub(crate) fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

pub(crate) fn mat3_mul_skew_right(a: [[f64; 3]; 3], v: [f64; 3]) -> [[f64; 3]; 3] {
    let x = v[0];
    let y = v[1];
    let z = v[2];
    [
        [
            a[0][1] * z - a[0][2] * y,
            -a[0][0] * z + a[0][2] * x,
            a[0][0] * y - a[0][1] * x,
        ],
        [
            a[1][1] * z - a[1][2] * y,
            -a[1][0] * z + a[1][2] * x,
            a[1][0] * y - a[1][1] * x,
        ],
        [
            a[2][1] * z - a[2][2] * y,
            -a[2][0] * z + a[2][2] * x,
            a[2][0] * y - a[2][1] * x,
        ],
    ]
}

pub(crate) fn mat3_transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

pub(crate) fn mat3_vec(a: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

pub(crate) fn mat3_from_flat(flat: &[f64], index: usize) -> [[f64; 3]; 3] {
    let start = index * 9;
    [
        [flat[start], flat[start + 1], flat[start + 2]],
        [flat[start + 3], flat[start + 4], flat[start + 5]],
        [flat[start + 6], flat[start + 7], flat[start + 8]],
    ]
}

pub(crate) fn mat6_vec(a: [[f64; 6]; 6], v: [[f64; 3]; 2]) -> [[f64; 3]; 2] {
    let x = [v[0][0], v[0][1], v[0][2], v[1][0], v[1][1], v[1][2]];
    let mut y = [0.0; 6];
    for r in 0..6 {
        y[r] = a[r][0] * x[0]
            + a[r][1] * x[1]
            + a[r][2] * x[2]
            + a[r][3] * x[3]
            + a[r][4] * x[4]
            + a[r][5] * x[5];
    }
    [[y[0], y[1], y[2]], [y[3], y[4], y[5]]]
}

pub(crate) fn set_mat3(flat: &mut [f64], index: usize, mat: [[f64; 3]; 3]) {
    let start = index * 9;
    for r in 0..3 {
        for c in 0..3 {
            flat[start + r * 3 + c] = mat[r][c];
        }
    }
}

pub(crate) fn mat3_col(flat: &[f64], index: usize, col: usize, cols: usize) -> [[f64; 3]; 3] {
    let start = index * 9 * cols + col;
    [
        [flat[start], flat[start + cols], flat[start + 2 * cols]],
        [
            flat[start + 3 * cols],
            flat[start + 4 * cols],
            flat[start + 5 * cols],
        ],
        [
            flat[start + 6 * cols],
            flat[start + 7 * cols],
            flat[start + 8 * cols],
        ],
    ]
}

pub(crate) fn set_mat3_col(
    flat: &mut [f64],
    index: usize,
    col: usize,
    cols: usize,
    mat: [[f64; 3]; 3],
) {
    let start = index * 9 * cols + col;
    for r in 0..3 {
        for c in 0..3 {
            flat[start + (r * 3 + c) * cols] = mat[r][c];
        }
    }
}

pub(crate) fn flat3(flat: &[f64], index: usize) -> [f64; 3] {
    let start = index * 3;
    [flat[start], flat[start + 1], flat[start + 2]]
}

pub(crate) fn set_flat3(flat: &mut [f64], index: usize, value: [f64; 3]) {
    let start = index * 3;
    flat[start] = value[0];
    flat[start + 1] = value[1];
    flat[start + 2] = value[2];
}

pub(crate) fn flat3_col(flat: &[f64], index: usize, col: usize, cols: usize) -> [f64; 3] {
    let start = index * 3 * cols + col;
    [flat[start], flat[start + cols], flat[start + 2 * cols]]
}

pub(crate) fn set_flat3_col(
    flat: &mut [f64],
    index: usize,
    col: usize,
    cols: usize,
    value: [f64; 3],
) {
    let start = index * 3 * cols + col;
    flat[start] = value[0];
    flat[start + cols] = value[1];
    flat[start + 2 * cols] = value[2];
}

pub(crate) fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn set_force(flat: &mut [f64], index: usize, torque: [f64; 3], force: [f64; 3]) {
    let start = index * 6;
    flat[start] = torque[0];
    flat[start + 1] = torque[1];
    flat[start + 2] = torque[2];
    flat[start + 3] = force[0];
    flat[start + 4] = force[1];
    flat[start + 5] = force[2];
}

pub(crate) fn force_torque(flat: &[f64], index: usize) -> [f64; 3] {
    let start = index * 6;
    [flat[start], flat[start + 1], flat[start + 2]]
}

pub(crate) fn force_force(flat: &[f64], index: usize) -> [f64; 3] {
    let start = index * 6;
    [flat[start + 3], flat[start + 4], flat[start + 5]]
}

pub(crate) fn set_force_col(
    flat: &mut [f64],
    index: usize,
    col: usize,
    cols: usize,
    torque: [f64; 3],
    force: [f64; 3],
) {
    let start = index * 6 * cols + col;
    flat[start] = torque[0];
    flat[start + cols] = torque[1];
    flat[start + 2 * cols] = torque[2];
    flat[start + 3 * cols] = force[0];
    flat[start + 4 * cols] = force[1];
    flat[start + 5 * cols] = force[2];
}

pub(crate) fn force_torque_col(flat: &[f64], index: usize, col: usize, cols: usize) -> [f64; 3] {
    let start = index * 6 * cols + col;
    [flat[start], flat[start + cols], flat[start + 2 * cols]]
}

pub(crate) fn force_force_col(flat: &[f64], index: usize, col: usize, cols: usize) -> [f64; 3] {
    let start = index * 6 * cols + col;
    [
        flat[start + 3 * cols],
        flat[start + 4 * cols],
        flat[start + 5 * cols],
    ]
}

pub(crate) fn add_shifted_force_parent(
    flat: &mut [f64],
    parent: usize,
    child: usize,
    rel: [f64; 3],
) {
    let p = parent * 6;
    let c = child * 6;
    let child_torque = [flat[c], flat[c + 1], flat[c + 2]];
    let child_force = [flat[c + 3], flat[c + 4], flat[c + 5]];
    let shifted_torque = add3(child_torque, cross(rel, child_force));
    flat[p] += shifted_torque[0];
    flat[p + 1] += shifted_torque[1];
    flat[p + 2] += shifted_torque[2];
    flat[p + 3] += child_force[0];
    flat[p + 4] += child_force[1];
    flat[p + 5] += child_force[2];
}

pub(crate) fn add_shifted_force_parent_derivative_col(
    dflat: &mut [f64],
    parent: usize,
    col: usize,
    cols: usize,
    rel: [f64; 3],
    drel: [f64; 3],
    child_force: [f64; 3],
    dchild_torque: [f64; 3],
    dchild_force: [f64; 3],
) {
    let p = parent * 6 * cols + col;
    let shifted_torque = add3(
        dchild_torque,
        add3(cross(drel, child_force), cross(rel, dchild_force)),
    );
    dflat[p] += shifted_torque[0];
    dflat[p + cols] += shifted_torque[1];
    dflat[p + 2 * cols] += shifted_torque[2];
    dflat[p + 3 * cols] += dchild_force[0];
    dflat[p + 4 * cols] += dchild_force[1];
    dflat[p + 5 * cols] += dchild_force[2];
}

pub(crate) fn set_tau_col(flat: &mut [f64], row: usize, col: usize, cols: usize, value: f64) {
    flat[row * cols + col] = value;
}

pub(crate) fn merge_columns(target: &mut Vec<usize>, source: &[usize]) {
    for &col in source {
        match target.binary_search(&col) {
            Ok(_) => {}
            Err(index) => target.insert(index, col),
        }
    }
}

pub(crate) fn set_jac(
    flat: &mut [f64],
    dof: usize,
    link: usize,
    row: usize,
    col: usize,
    value: f64,
) {
    flat[(link * 6 + row) * dof + col] = value;
}
