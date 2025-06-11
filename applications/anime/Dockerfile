# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} AS run

ARG DEBIAN_FRONTEND=noninteractive

# Install only minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python

#
# Holohub run setup
#
RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup

WORKDIR /app/anime
RUN wget https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Hayao_36.onnx
RUN wget https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Shinkai_37.onnx

RUN pip install onnxconverter-common onnxmltools
COPY converter.py /tmp
RUN python3 /tmp/converter.py
