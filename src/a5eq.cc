/*
 * Copyright (C) 2023 Ayan Shafqat <ayan.x.shafqat@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdlib>
#include <memory>

#if _WIN32
#include "lv2plug.in/ns/lv2core/lv2.h"
#else
#include "lv2/lv2plug.in/ns/lv2core/lv2.h"
#endif

#include "eq.hpp"

#ifndef LV2_SYMBOL_EXPORT
#ifdef _WIN32
#define LV2_SYMBOL_EXPORT __declspec(dllexport)
#else
#define LV2_SYMBOL_EXPORT __attribute__((visibility("default")))
#endif
#endif // LV2_SYMBOL_EXPORT

#define A5EQ_URI "urn:ashafq:a5eq"

/**
 * LV2 API functions
 **/

const LV2_Descriptor *lv2_descriptor(uint32_t index);

static LV2_Handle instantiate(const LV2_Descriptor *descriptor, double rate,
                              const char *bundle_path,
                              const LV2_Feature *const *features);

static void connect(LV2_Handle instance, uint32_t port, void *data);

static void run(LV2_Handle instance, uint32_t frames);

static void cleanup(LV2_Handle instance);

static const void *extension_data(const char *uri);

/**
 * LV2 plugin description
 **/
static const LV2_Descriptor descriptor_mono = {
    A5EQ_URI "#mono",
    instantiate,
    connect,
    nullptr,
    run,
    nullptr,
    cleanup,
    extension_data
};

/**
 * LV2 plugin description
 **/
static const LV2_Descriptor descriptor_stereo = {
    A5EQ_URI "#stereo",
    instantiate,
    connect,
    nullptr,
    run,
    nullptr,
    cleanup,
    extension_data
};


LV2_SYMBOL_EXPORT
const LV2_Descriptor *lv2_descriptor(uint32_t index) {
  switch (index) {
  case 0:
    return &descriptor_mono;
  case 1:
    return &descriptor_stereo;
  default:
    return nullptr;
  }
}

/**
 * LV2 API function implementation
 **/

static LV2_Handle instantiate(const LV2_Descriptor *descriptor, double rate,
                              const char *bundle_path,
                              const LV2_Feature *const *features) {
  ((void)descriptor);
  ((void)bundle_path);
  ((void)features);

  void *ptr = malloc(sizeof(A5Eq));
  A5Eq *self = new (ptr) A5Eq();
  self->set_sample_rate(rate);

  return (LV2_Handle)(self);
}

static void connect(LV2_Handle instance, uint32_t port, void *data) {
  A5Eq *self = reinterpret_cast<A5Eq *>(instance);
  self->connect_port(port, data);
}

static void run(LV2_Handle instance, uint32_t frames) {
  A5Eq *self = reinterpret_cast<A5Eq *>(instance);
  self->process(frames);
}

static void cleanup(LV2_Handle instance) { free(instance); }

static const void *extension_data(const char *uri) {
  ((void)(uri));
  return nullptr;
}
