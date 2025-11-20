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
project(anime NONE)

message(STATUS "Starting anime deployment setup...")

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


# Download the testing video and trim
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/anime/running_on_pathway.mp4"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/anime"
    COMMAND curl -S -o "${HOLOHUB_DATA_DIR}/anime/video.mp4"
              -L "https://www.pexels.com/download/video/5823544/?fps=50.0&h=720&w=1280"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/anime/video.mp4" -t 6.4
              -c:v libx264 "${HOLOHUB_DATA_DIR}/anime/running_on_pathway.mp4"
    COMMAND rm -rf "${HOLOHUB_DATA_DIR}/anime/video.mp4"
    VERBATIM
    )


# Convert the video file
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/anime/running_on_pathway.gxf_index"
                          "${HOLOHUB_DATA_DIR}/anime/running_on_pathway.gxf_entities"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/anime/running_on_pathway.mp4" -pix_fmt rgb24 -f rawvideo pipe:1 |
            python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
            --directory "${HOLOHUB_DATA_DIR}/anime" --basename running_on_pathway --width 1280 --height 720 --framerate 50
    DEPENDS "${HOLOHUB_DATA_DIR}/anime/running_on_pathway.mp4"
    )

add_custom_target(depth_anything_v2_data ALL
                  DEPENDS
                  "${HOLOHUB_DATA_DIR}/anime/running_on_pathway.gxf_index")
