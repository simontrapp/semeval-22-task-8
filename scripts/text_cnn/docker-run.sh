#!/bin/bash

set -e

# in /home/stud/trapp, run:
#   git clone --branch simon https://gitlab2.informatik.uni-wuerzburg.de/s364758/mlnlp.git
#   cd mlnlp/data/processed/
#   unzip -q training_data.zip
#   mv training_data train
#   cd ../../
#   chmod +x scripts/bert_sdr/docker_run.sh
#   ./scripts/bert_sdr/docker_run.sh

export BUILDAH_FORMAT="docker"
export NAME="ls6-stud-registry.informatik.uni-wuerzburg.de/studheinickel/sim-cnn:0.0.1"
alias buildah='buildah --runroot /tmp/$USER/.local/share/containers/runroot --root /tmp/$USER/.local/share/containers/storage/'

echo "Building the container..."
buildah bud -t ${NAME} -f scripts/text_cnn/Dockerfile.update .
echo "Login to container registry. Username: stud, Password: studregistry."
buildah login ls6-stud-registry.informatik.uni-wuerzburg.de   # with username `stud` and password `studregistry`
echo "Pushing container to registry..."
buildah push ${NAME}

# RUN ONCE: kubectl -n studheinickel create secret generic lsx-registry --from-file=.dockerconfigjson=${XDG_RUNTIME_DIR}/containers/auth.json --type=kubernetes.io/dockerconfigjson

kubectl -n studheinickel create -f scripts/text_cnn/docker_run.yaml

# TO DELETE: kubectl -n studheinickel delete -f scripts/bert_sdr/docker_run.yaml

