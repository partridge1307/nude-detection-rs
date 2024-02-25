use opencv::core::Rect_;

pub type Rect = Rect_<i32>;

#[derive(Debug, PartialEq, Eq)]
pub enum Label {
    FemaleGenitaliaCovered,
    FaceFemale,
    ButtocksExposed,
    FemaleBreastExposed,
    FemaleGenitaliaExposed,
    MaleBreastExposed,
    AnusExposed,
    FeetExposed,
    BellyCovered,
    FeetCovered,
    ArmpitsCovered,
    ArmpitsExposed,
    FaceMale,
    BellyExposed,
    MaleGenitaliaExposed,
    AnusCovered,
    FemaleBreastCovered,
    ButtocksCovered,
}

#[derive(Debug)]
pub struct Detection {
    pub class: Label,
    pub score: f32,
    pub rect: Rect,
}

impl TryFrom<usize> for Label {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Label::FemaleGenitaliaCovered),
            1 => Ok(Label::FaceFemale),
            2 => Ok(Label::ButtocksExposed),
            3 => Ok(Label::FemaleBreastExposed),
            4 => Ok(Label::FemaleGenitaliaExposed),
            5 => Ok(Label::MaleBreastExposed),
            6 => Ok(Label::AnusExposed),
            7 => Ok(Label::FeetExposed),
            8 => Ok(Label::BellyCovered),
            9 => Ok(Label::FeetCovered),
            10 => Ok(Label::ArmpitsCovered),
            11 => Ok(Label::ArmpitsExposed),
            12 => Ok(Label::FaceMale),
            13 => Ok(Label::BellyExposed),
            14 => Ok(Label::MaleGenitaliaExposed),
            15 => Ok(Label::AnusCovered),
            16 => Ok(Label::FemaleBreastCovered),
            17 => Ok(Label::ButtocksCovered),
            _ => Err("Invalid class"),
        }
    }
}
