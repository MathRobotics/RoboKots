//! Canonical model decoding. Parsing and validation need neither Python nor NumPy.
use crate::error::{CoreResult, Error};
use crate::model::{JointKind, JointModel, LinkModel, RobotModel};
use crate::types::RustCompiledRobot;
use serde_json::{Map, Value};

fn object<'a>(v: &'a Value, field: &str) -> CoreResult<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| Error::new(format!("{field} must be an object")))
}
fn required<'a>(v: &'a Map<String, Value>, key: &str) -> CoreResult<&'a Value> {
    v.get(key)
        .ok_or_else(|| Error::new(format!("missing key: {key}")))
}
fn string<'a>(v: &'a Value, key: &str) -> CoreResult<&'a str> {
    v.as_str()
        .ok_or_else(|| Error::new(format!("{key} must be a string")))
}
fn index(v: &Value, key: &str) -> CoreResult<usize> {
    v.as_u64()
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| Error::new(format!("{key} must be a nonnegative integer")))
}
fn number(v: &Value, key: &str) -> CoreResult<f64> {
    v.as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| Error::new(format!("{key} must be a finite number")))
}
fn vector<const N: usize>(v: Option<&Value>, default: [f64; N], key: &str) -> CoreResult<[f64; N]> {
    let Some(v) = v else {
        return Ok(default);
    };
    let a = v
        .as_array()
        .filter(|a| a.len() == N)
        .ok_or_else(|| Error::new(format!("{key} must have length {N}")))?;
    let mut out = default;
    for (dst, src) in out.iter_mut().zip(a) {
        *dst = number(src, key)?;
    }
    Ok(out)
}
fn sorted_items<'a>(
    data: &'a Map<String, Value>,
    key: &str,
) -> CoreResult<Vec<&'a Map<String, Value>>> {
    let values = required(data, key)?
        .as_array()
        .ok_or_else(|| Error::new(format!("{key} must be an array")))?;
    let mut items = values
        .iter()
        .map(|v| {
            let item = object(v, key)?;
            Ok((index(required(item, "id")?, "id")?, item))
        })
        .collect::<CoreResult<Vec<_>>>()?;
    items.sort_by_key(|x| x.0);
    if items.iter().enumerate().any(|(i, x)| i != x.0) {
        return Err(Error::new(format!(
            "{key}.id must be unique contiguous integers starting at zero"
        )));
    }
    Ok(items.into_iter().map(|x| x.1).collect())
}

impl RobotModel {
    /// Decode schema 0.0.2, preserving motion order by sorting owners by ID.
    /// Compilation subsequently validates names, numerical values and tree topology.
    pub fn from_json(text: &str) -> CoreResult<Self> {
        let value = serde_json::from_str(text)
            .map_err(|e| Error::new(format!("invalid model JSON: {e}")))?;
        Self::from_json_value(&value)
    }

    pub fn from_json_value(value: &Value) -> CoreResult<Self> {
        let data = object(value, "model")?;
        if required(data, "schema_version")?.as_str() != Some("0.0.2") {
            return Err(Error::new("schema_version must be '0.0.2'"));
        }
        let links = sorted_items(data, "links")?
            .into_iter()
            .map(|link| {
                if link
                    .get("type")
                    .map(|v| v.as_str() != Some("rigid"))
                    .unwrap_or(false)
                {
                    return Err(Error::new("Rust model input supports rigid links only"));
                }
                let mut inertia = [1., 1., 1., 0., 0., 0.];
                if let Some(v) = link.get("inertia").filter(|v| !v.is_null()) {
                    let fields = object(v, "inertia")?;
                    let keys = ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"];
                    if fields.len() != keys.len() {
                        return Err(Error::new(
                            "inertia must contain exactly ixx, iyy, izz, ixy, ixz, iyz",
                        ));
                    }
                    for (i, key) in keys.iter().enumerate() {
                        inertia[i] = number(required(fields, key)?, key)?;
                    }
                }
                Ok(LinkModel {
                    name: string(required(link, "name")?, "name")?.into(),
                    mass: link
                        .get("mass")
                        .map(|v| number(v, "mass"))
                        .transpose()?
                        .unwrap_or(0.),
                    cog: vector(link.get("cog"), [0.; 3], "cog")?,
                    inertia,
                })
            })
            .collect::<CoreResult<Vec<_>>>()?;
        let joints = sorted_items(data, "joints")?
            .into_iter()
            .map(|joint| {
                let kind = string(required(joint, "type")?, "type")?;
                let representation = joint
                    .get("q_representation")
                    .filter(|v| !v.is_null())
                    .map(|v| string(v, "q_representation"))
                    .transpose()?
                    .unwrap_or("");
                if !["", "rotation_vector", "expmap"].contains(&representation) {
                    return Err(Error::new("unsupported q_representation"));
                }
                let kind = JointKind::parse(kind, representation)?;
                let dof = usize::from(kind != JointKind::Fixed);
                if let Some(v) = joint.get("dof") {
                    if index(v, "dof")? != dof {
                        return Err(Error::new("joint dof does not match its type"));
                    }
                }
                let empty = Map::new();
                let origin = joint
                    .get("origin")
                    .map(|v| object(v, "origin"))
                    .transpose()?
                    .unwrap_or(&empty);
                if kind != JointKind::Fixed {
                    required(joint, "axis")?;
                }
                Ok(JointModel {
                    name: string(required(joint, "name")?, "name")?.into(),
                    kind,
                    parent_link: index(required(joint, "parent_link_id")?, "parent_link_id")?,
                    child_link: index(required(joint, "child_link_id")?, "child_link_id")?,
                    axis: vector(joint.get("axis"), [1., 0., 0.], "axis")?,
                    position: vector(origin.get("position"), [0.; 3], "position")?,
                    orientation: vector(
                        origin.get("orientation"),
                        [1., 0., 0., 0.],
                        "orientation",
                    )?,
                })
            })
            .collect::<CoreResult<Vec<_>>>()?;
        Ok(Self { links, joints })
    }
}

impl RustCompiledRobot {
    pub fn from_json(text: &str, allow_prismatic: bool) -> CoreResult<Self> {
        Self::from_model(&RobotModel::from_json(text)?, allow_prismatic)
    }

    pub fn from_json_file(
        path: impl AsRef<std::path::Path>,
        allow_prismatic: bool,
    ) -> CoreResult<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::new(format!("cannot read model: {e}")))?;
        Self::from_json(&text, allow_prismatic)
    }
}
