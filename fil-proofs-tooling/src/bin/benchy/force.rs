use fil_proofs_tooling::shared::{PROVER_ID, TICKET_BYTES};
use std::{
    fs::OpenOptions,
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
};

use fil_proofs_tooling::{measure, FuncMeasurement};
use filecoin_proofs::{
    add_piece, generate_piece_commitment, seal_pre_commit_phase1, with_shape, MerkleTreeTrait,
    PaddedBytesAmount, PoRepConfig, PoRepProofPartitions, SealPreCommitPhase1Output, SectorSize,
    UnpaddedBytesAmount, POREP_PARTITIONS,
};
use log::info;
use storage_proofs_core::{api_version::ApiVersion, sector::SectorId};

const SECTOR_ID: u64 = 0;

const PIECE_FILE: &str = "piece-file";
const PIECE_INFOS_FILE: &str = "piece-infos-file";
const STAGED_FILE: &str = "staged-file";
const SEALED_FILE: &str = "sealed-file";
const PRECOMMIT_PHASE1_OUTPUT_FILE: &str = "precommit-phase1-output";
const PRECOMMIT_PHASE2_OUTPUT_FILE: &str = "precommit-phase2-output";
const COMMIT_PHASE1_OUTPUT_FILE: &str = "commit-phase1-output";

pub fn run_pc1(sector_size: u64, cache_dir: PathBuf) -> anyhow::Result<()> {
    with_shape!(sector_size, run_pc1_inner, sector_size, cache_dir)
}

pub fn run_pc1_inner<Tree: 'static + MerkleTreeTrait>(
    sector_size: u64,
    cache_dir: PathBuf,
) -> anyhow::Result<()> {

    // Create files for the staged and sealed sectors.
    let staged_file_path = cache_dir.join(STAGED_FILE);
    info!("*** Creating staged file");
    let mut staged_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&staged_file_path)?;

    let sealed_file_path = cache_dir.join(SEALED_FILE);
    info!("*** Creating sealed file");
    let _ = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&sealed_file_path);

    let sector_size_unpadded_bytes_amount =
        UnpaddedBytesAmount::from(PaddedBytesAmount(sector_size));

    let piece_file_path = cache_dir.join(PIECE_FILE);
    // Generate the data from which we will create a replica, we will then prove the continued
    // storage of that replica using the PoSt.
    let piece_bytes: Vec<u8> = (0..usize::from(sector_size_unpadded_bytes_amount))
        .map(|_| rand::random::<u8>())
        .collect();

    info!("*** Created piece file");
    let mut piece_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&piece_file_path)?;
    piece_file.write_all(&piece_bytes)?;
    piece_file.sync_all()?;
    piece_file.seek(SeekFrom::Start(0))?;

    let piece_info = generate_piece_commitment(&mut piece_file, sector_size_unpadded_bytes_amount)?;
    piece_file.seek(SeekFrom::Start(0))?;

    add_piece(
        &mut piece_file,
        &mut staged_file,
        sector_size_unpadded_bytes_amount,
        &[],
    )?;

    let piece_infos = vec![piece_info];
    let sector_id = SectorId::from(SECTOR_ID);
    let porep_config = get_porep_config(sector_size, ApiVersion::V1_1_0);

    let seal_pre_commit_phase1_measurement: FuncMeasurement<SealPreCommitPhase1Output<Tree>> =
        measure(|| {
            seal_pre_commit_phase1::<_, _, _, Tree>(
                porep_config,
                cache_dir.clone(),
                staged_file_path.clone(),
                sealed_file_path.clone(),
                PROVER_ID,
                sector_id,
                TICKET_BYTES,
                &piece_infos,
            )
        })
        .expect("failed in seal_pre_commit_phase1");
    info!(
        "cpu_time: {:?}, will_time: {:?}",
        seal_pre_commit_phase1_measurement.cpu_time, seal_pre_commit_phase1_measurement.wall_time
    );
    println!(
        "pc1out: {:?}",
        seal_pre_commit_phase1_measurement.return_value
    );

    Ok(())
}

fn get_porep_config(sector_size: u64, api_version: ApiVersion) -> PoRepConfig {
    let arbitrary_porep_id = [99; 32];

    // Replicate the staged sector, write the replica file to `sealed_path`.
    PoRepConfig {
        sector_size: SectorSize(sector_size),
        partitions: PoRepProofPartitions(
            *POREP_PARTITIONS
                .read()
                .expect("POREP_PARTITONS poisoned")
                .get(&(sector_size))
                .expect("unknown sector size"),
        ),
        porep_id: arbitrary_porep_id,
        api_version,
    }
}
