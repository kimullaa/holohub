# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# target does nothing but be created if this folder is reached
# to set dependencies on adequate operators
# add_custom_target(advanced_networking_bench_python ALL
#   DEPENDS advanced_network_python
# )

install(
  FILES main.py
  DESTINATION examples/adv_networking_bench/
  RENAME adv_networking_bench.py
  COMPONENT adv_networking_bench-py
)
