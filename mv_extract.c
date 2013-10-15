// Based on a code by FFmpeg and Victor Hsieh.

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <jansson.h>

#define IS_INTERLACED(a) ((a)&MB_TYPE_INTERLACED)
#define IS_16X16(a)      ((a)&MB_TYPE_16x16)
#define IS_16X8(a)       ((a)&MB_TYPE_16x8)
#define IS_8X16(a)       ((a)&MB_TYPE_8x16)
#define IS_8X8(a)        ((a)&MB_TYPE_8x8)
#define USES_LIST(a, list) ((a) & ((MB_TYPE_P0L0|MB_TYPE_P1L0)<<(2*(list))))

#define NO_MV 10000

int count = 0;
double sum = 0;
double w;
double h;

void print_vector(int x, int y, int dx, int dy)
{
  if (dx != NO_MV && dy != NO_MV){
    sum = sum + sqrt((double)dx*(double)dx/w/w + (double)dy*(double)dy/h/h);
    count++;
  }
    //printf("%d %d ; %d %d\n", x, y, dx, dy);
}

/* Print motion vector for each macroblock in this frame.  If there is
 * no motion vector in some macroblock, it prints a magic number NO_MV. */
void printMVMatrix(int index, AVFrame *pict, AVCodecContext *ctx)
{
  const int mb_width  = (ctx->width + 15) / 16;
  const int mb_height = (ctx->height + 15) / 16;
  const int mb_stride = mb_width + 1;
  const int mv_sample_log2 = 4 - pict->motion_subsample_log2;
  const int mv_stride = (mb_width << mv_sample_log2) + (ctx->codec_id == CODEC_ID_H264 ? 0 : 1);
  const int quarter_sample = (ctx->flags & CODEC_FLAG_QPEL) != 0;
  const int shift = 1 + quarter_sample;
  int mb_y, mb_x, type, i;


  printf("frame %d, %d x %d\n", index, mb_height, mb_width);

  for (mb_y = 0; mb_y < mb_height; mb_y++) {
    for (mb_x = 0; mb_x < mb_width; mb_x++) {
      sum = 0;
      const int mb_index = mb_x + mb_y * mb_stride;
      if (pict->motion_val) {
        for (type = 0; type < 3; type++) {
          int direction = 0;
          switch (type) {
            case 0:
            if (pict->pict_type != AV_PICTURE_TYPE_P)
              continue;
            direction = 0;
            break;
            case 1:
            if (pict->pict_type != AV_PICTURE_TYPE_B)
              continue;
            direction = 0;
            break;
            case 2:
            if (pict->pict_type != AV_PICTURE_TYPE_B)
              continue;
            direction = 1;
            break;
          }

          if (!USES_LIST(pict->mb_type[mb_index], direction)) {
            if (IS_8X8(pict->mb_type[mb_index])) {
              print_vector(mb_x, mb_y, NO_MV, NO_MV);
              print_vector(mb_x, mb_y, NO_MV, NO_MV);
              print_vector(mb_x, mb_y, NO_MV, NO_MV);
              print_vector(mb_x, mb_y, NO_MV, NO_MV);
            } else if (IS_16X8(pict->mb_type[mb_index])) {
              print_vector(mb_x, mb_y, NO_MV, NO_MV);
              print_vector(mb_x, mb_y, NO_MV, NO_MV);
            } else if (IS_8X16(pict->mb_type[mb_index])) {
              print_vector(mb_x, mb_y, NO_MV, NO_MV);
              print_vector(mb_x, mb_y, NO_MV, NO_MV);
            } else {
              print_vector(mb_x, mb_y, NO_MV, NO_MV);
            }
            continue;
          }

          if (IS_8X8(pict->mb_type[mb_index])) {
            for (i = 0; i < 4; i++) {
              int xy = (mb_x*2 + (i&1) + (mb_y*2 + (i>>1))*mv_stride) << (mv_sample_log2-1);
              int dx = (pict->motion_val[direction][xy][0]>>shift);
              int dy = (pict->motion_val[direction][xy][1]>>shift);
              print_vector(mb_x, mb_y, dx, dy);
            }
          } else if (IS_16X8(pict->mb_type[mb_index])) {
            for (i = 0; i < 2; i++) {
              int xy = (mb_x*2 + (mb_y*2 + i)*mv_stride) << (mv_sample_log2-1);
              int dx = (pict->motion_val[direction][xy][0]>>shift);
              int dy = (pict->motion_val[direction][xy][1]>>shift);

              if (IS_INTERLACED(pict->mb_type[mb_index]))
                dy *= 2;

              print_vector(mb_x, mb_y, dx, dy);
            }
          } else if (IS_8X16(pict->mb_type[mb_index])) {
            for (i = 0; i < 2; i++) {
              int xy =  (mb_x*2 + i + mb_y*2*mv_stride) << (mv_sample_log2-1);
              int dx = (pict->motion_val[direction][xy][0]>>shift);
              int dy = (pict->motion_val[direction][xy][1]>>shift);

              if (IS_INTERLACED(pict->mb_type[mb_index]))
                dy *= 2;

              print_vector(mb_x, mb_y, dx, dy);
            }
          } else {
            int xy = (mb_x + mb_y*mv_stride) << mv_sample_log2;
            int dx = (pict->motion_val[direction][xy][0]>>shift);
            int dy = (pict->motion_val[direction][xy][1]>>shift);
            print_vector(mb_x, mb_y, dx, dy);
          }
        }

        if (sum > 0.06) printf("# ");
        else if (sum > 0) printf("- ");
        else printf("  ");
      }
    }
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  AVFormatContext *pFormatCtx;
  int i, videoStream;
  AVCodecContext *pCodecCtx;
  AVCodec *pCodec;
  AVFrame *pFrame;
  AVPacket packet;
  int frameFinished;
  int numBytes;
  uint8_t *buffer;

  json_t *frames = json_array();
  json_error_t error;

  if (argc < 2) {
    printf("Please provide a movie file\n");
    return -1;
  }

  // Register all formats and codecs
  av_register_all();

  // Open video file
  if (avformat_open_input(&pFormatCtx, argv[1], NULL, NULL) != 0)
    return -1; // Couldn't open file

  // Retrieve stream information
  if (avformat_find_stream_info(pFormatCtx, NULL) < 0) 
    return -1; // Couldn't find stream information

  // Dump information about file onto standard error
  av_dump_format(pFormatCtx, 0, argv[1], 0);

  // Find the first video stream
  videoStream = -1;
  for (i = 0; i < pFormatCtx->nb_streams; i++) {
    if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoStream = i;
      break;
    }
  }

  if (videoStream == -1)
    return -1; // Didn't find a video stream

  // Get a pointer to the codec context for the video stream
  pCodecCtx = pFormatCtx->streams[videoStream]->codec;

  // Find the decoder for the video stream
  pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
  if (pCodec == NULL) {
    fprintf(stderr, "Unsupported codec!\n");
    return -1; // Codec not found
  }
  
  // Open codec
  if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0)
    return -1; // Could not open codec

  // Allocate video frame
  pFrame = avcodec_alloc_frame();

  w = pCodecCtx->width;
  h = pCodecCtx->height;

  // Read frames and save first five frames to disk
  i = 0;
  while (av_read_frame(pFormatCtx, &packet) >= 0)
  {
    // Is this a packet from the video stream?
    if (packet.stream_index == videoStream)
    {
        // Decode video frame
      avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);

        // Did we get a video frame?
      if (frameFinished)
      {
        if (pFrame->pict_type != AV_PICTURE_TYPE_I)
          printMVMatrix(i++, pFrame, pCodecCtx);
      }
    }

    // Free the packet that was allocated by av_read_frame
    av_free_packet(&packet);
  }

  // Free the YUV frame
  av_free(pFrame);

  // Close the codec
  avcodec_close(pCodecCtx);

  // Close the video file
  avformat_close_input(&pFormatCtx);

  return 0;
}
