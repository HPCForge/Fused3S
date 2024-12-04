#ifndef CONFIG_H
#define CONFIG_H

#define BLK_8 8

#define BLK_H 16 
#define BLK_W 8

#define BLK_M 16
#define BLK_N 8
#define BLK_K 16

#define WARP_SIZE 32
// each warp does 2 TCBlocks
#define TCBLOCK_PER_WARP 2
// #define TCBLOCK_PER_WARP 128 
#define TCBLOCK_PER_WARP_FMM 2
// #define TCBLOCK_PER_WARP 32
#endif