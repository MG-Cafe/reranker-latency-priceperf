#!/bin/bash
# Provision a single-GPU G4 (1x NVIDIA RTX PRO 6000) VM on GCP for the reranker latency benchmark.
# Uses machine type g4-standard-48 (1 GPU) and the Ubuntu 24.04 accelerator image so the GPU driver
# initializes. Set your own GCP project id (or export PROJECT_ID before running). No credentials here.
P="${PROJECT_ID:-your-project-id}"; NAME="g4-bench"; MT="g4-standard-48"
IMG_FAMILY="ubuntu-accelerator-2404-amd64-with-nvidia-570"; IMG_PROJECT="ubuntu-os-accelerator-images"
ZONES="us-central1-b us-west1-a us-west1-b us-west1-c us-east1-b us-east1-d europe-west1-c asia-northeast1-b asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-south1-c asia-east1-a europe-north1-a"
LOG="$HOME/g4_provision.log"; echo "start $(date -u +%FT%TZ)" > "$LOG"
try(){ local extra="$1"; for Z in $ZONES; do
  OUT=$(gcloud compute instances create "$NAME" --project="$P" --zone="$Z" --machine-type="$MT" $extra \
    --image-family="$IMG_FAMILY" --image-project="$IMG_PROJECT" --boot-disk-size=200GB \
    --boot-disk-type=hyperdisk-balanced --maintenance-policy=TERMINATE --metadata=enable-oslogin=FALSE 2>&1)
  if echo "$OUT" | grep -q "RUNNING"; then echo "CREATED_IN=$Z"; echo "CREATED_IN=$Z $extra" >>"$LOG"; return 0; fi
  echo "$(date -u +%FT%TZ) $Z fail" >>"$LOG"
done; return 1; }
echo "== SPOT ==" >>"$LOG"; if try "--provisioning-model=SPOT --instance-termination-action=DELETE"; then exit 0; fi
echo "== ON-DEMAND ==" >>"$LOG"; if try ""; then exit 0; fi
echo NO_CAPACITY; echo NO_CAPACITY >>"$LOG"; exit 1
