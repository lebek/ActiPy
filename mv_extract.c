// Based on a code by FFmpeg and Victor Hsieh.

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <jansson.h>

#define IS_INTERLACED(a) ((a)&MB_TYPE_INTERLACED)
#define IS_16X16(a)      ((a)&MB_TYPE_16x16)
#define IS_16X8(a)       ((a)&MB_TYPE_16x8)
#define IS_8X16(a)       ((a)&MB_TYPE_8x16)
#define IS_8X8(a)        ((a)&MB_TYPE_8x8)
#define USES_LIST(a, list) ((a) & ((MB_TYPE_P0L0|MB_TYPE_P1L0)<<(2*(list))))

#define PIXELS_TO_MBS(p) ((int)((p + 15) / 16))
//#define PIXELS_TO_MBS(p) (int)p

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

/* 
 * Extracts the motion vector for each macroblock in a given frame.
 * For macroblock at (x,y), places the average x-velocity in outMatrix[0][x][y],
 * and the average y-velocity in outMatrix[1][x][y].
 */
void extractMVMatrix(int index, AVFrame *pict, AVCodecContext *ctx, int *(*(*outMatrix)))
{
  const int mb_width  = PIXELS_TO_MBS(ctx->width);
  const int mb_height = PIXELS_TO_MBS(ctx->height);
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
      outMatrix[0][mb_x][mb_y] = 0;
      outMatrix[1][mb_x][mb_y] = 0;

      const int mb_index = mb_x + mb_y * mb_stride;
      if (pict->motion_val) {
        for (type = 0; type < 2; type++) {
          int direction = 0;
          int divisor = 1;
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
            divisor = 2;
            break;
            case 2:
            if (pict->pict_type != AV_PICTURE_TYPE_B)
              continue;
            direction = 1;
            divisor = 2;
            break;
          }

#define SET_XMV(dx) outMatrix[0][mb_x][mb_y] = outMatrix[0][mb_x][mb_y] + dx
#define SET_YMV(dy) outMatrix[1][mb_x][mb_y] = outMatrix[1][mb_x][mb_y] + dy
          /* No motion vectors, just assume zero... */
          if (!USES_LIST(pict->mb_type[mb_index], direction)) {
            SET_XMV(0);
            SET_YMV(0);
            continue;
          }

          if (IS_8X8(pict->mb_type[mb_index])) {
            for (i = 0; i < 4; i++) {
              int xy = (mb_x*2 + (i&1) + (mb_y*2 + (i>>1))*mv_stride) << (mv_sample_log2-1);
              int dx = (pict->motion_val[direction][xy][0]>>shift);
              int dy = (pict->motion_val[direction][xy][1]>>shift);

              SET_XMV(dx);
              SET_YMV(dy);
            }
          } else if (IS_16X8(pict->mb_type[mb_index])) {
            for (i = 0; i < 2; i++) {
              int xy = (mb_x*2 + (mb_y*2 + i)*mv_stride) << (mv_sample_log2-1);
              int dx = (pict->motion_val[direction][xy][0]>>shift);
              int dy = (pict->motion_val[direction][xy][1]>>shift);

              if (IS_INTERLACED(pict->mb_type[mb_index]))
                dy *= 2;

              SET_XMV(dx);
              SET_YMV(dy);
            }
          } else if (IS_8X16(pict->mb_type[mb_index])) {
            for (i = 0; i < 2; i++) {
              int xy =  (mb_x*2 + i + mb_y*2*mv_stride) << (mv_sample_log2-1);
              int dx = (pict->motion_val[direction][xy][0]>>shift);
              int dy = (pict->motion_val[direction][xy][1]>>shift);

              if (IS_INTERLACED(pict->mb_type[mb_index]))
                dy *= 2;

              SET_XMV(dx);
              SET_YMV(dy);
            }
          } else {
            int xy = (mb_x + mb_y*mv_stride) << mv_sample_log2;
            int dx = (pict->motion_val[direction][xy][0]>>shift);
            int dy = (pict->motion_val[direction][xy][1]>>shift);
            
            SET_XMV(dx);
            SET_YMV(dy);
          }
#undef SET_XMV
#undef SET_YMV
        }
      }
    }
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

  json_t *root = json_object();
  json_t *frames = json_array();
  json_error_t error;

  json_object_set(root, "frames", frames);

  if (argc < 2) {
    printf("Please provide a movie file\n");
    return -1;
  }

  json_object_set_new(root, "file", json_string(argv[1]));

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

  json_object_set_new(root, "width", json_integer(w));
  json_object_set_new(root, "height", json_integer(h));

  int **x_mvs = (int **) malloc(PIXELS_TO_MBS(w) * sizeof(int *));
  int **y_mvs = (int **) malloc(PIXELS_TO_MBS(w) * sizeof(int *));
  for (int i = 0; i < PIXELS_TO_MBS(w); i++) {
   x_mvs[i] = (int *) malloc(PIXELS_TO_MBS(h) * sizeof(int));
   memset(x_mvs[i], 0, PIXELS_TO_MBS(h));
   y_mvs[i] = (int *) malloc(PIXELS_TO_MBS(h) * sizeof(int));
   memset(y_mvs[i], 0, PIXELS_TO_MBS(h));
  }
  int **mvs[] = { x_mvs, y_mvs };

  // Read frames and save first five frames to disk
  i = 0;
  while (av_read_frame(pFormatCtx, &packet) >= 0) {
    // Is this a packet from the video stream?
    if (packet.stream_index == videoStream) {
      
      // Decode video frame
      avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);

      // Did we get a video frame?
      if (frameFinished) {
        if (pFrame->pict_type != AV_PICTURE_TYPE_I) {
          extractMVMatrix(i++, pFrame, pCodecCtx, mvs);
          for (int k = 0; k < PIXELS_TO_MBS(h); k++) {
            for (int j = 0; j < PIXELS_TO_MBS(w); j++) {
              sum = 0;
              print_vector(k, j, mvs[0][j][k], mvs[1][j][k]);
              if (sum > 0.06) printf("# ");
              else if (sum > 0) printf("- ");
              else printf("  ");
            }
            printf("\n");
          }

          json_t *frame = json_object();
          json_array_append(frames, frame);

          json_t *frame_x_mvs = json_array();
          json_object_set(frame, "x", frame_x_mvs);

          json_t *frame_y_mvs = json_array();
          json_object_set(frame, "y", frame_y_mvs);

          for (int k = 0; k < PIXELS_TO_MBS(w); k++) {
            json_t *frame_x_mvs_col = json_array();
            json_array_append(frame_x_mvs, frame_x_mvs_col);

            json_t *frame_y_mvs_col = json_array();
            json_array_append(frame_y_mvs, frame_y_mvs_col);

            for (int j = 0; j < PIXELS_TO_MBS(h); j++) {
              json_array_append_new(frame_x_mvs_col, json_integer(x_mvs[k][j]));
              x_mvs[k][j] = 0;

              json_array_append_new(frame_y_mvs_col, json_integer(y_mvs[k][j]));
              y_mvs[k][j] = 0;
            }
          }
        }
      }
    }

    // Free the packet that was allocated by av_read_frame
    av_free_packet(&packet);
  }

  json_dump_file(root, "mv_dump.json", 0);

  free(x_mvs);
  free(y_mvs);

  // Free the YUV frame
  av_free(pFrame);

  // Close the codec
  avcodec_close(pCodecCtx);

  // Close the video file
  avformat_close_input(&pFormatCtx);

  return 0;
}
