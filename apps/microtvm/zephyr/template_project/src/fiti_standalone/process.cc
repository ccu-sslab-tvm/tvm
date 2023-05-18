#include "process.h"

#include <stdio.h>
#include <cstdarg>
#include <vector>
#include <algorithm>

#define PIC_SIZE		192
#define CLASS_NUM 		5
#define CANDIDATE		540
#define CONFI_THRESHOLD	0.3
#define SCORE_THRESHOLD	0.4
#define IOU_THRESHOLD	0.3
#define DEQNT(num)		float(0.006245302967727184 * (num + 122))

using namespace std;

void TVMLogf(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  vfprintf(stderr, msg, args);
  va_end(args);
}

struct Bbox {
	int x1, y1, x2, y2;
	float confidence;
	int class_id;

	Bbox(int _x1, int _y1, int _x2, int _y2, float _score, int _id) :
		x1(_x1), y1(_y1), x2(_x2), y2(_y2), confidence(_score), class_id(_id) {}
};

int clamp(int num, int min, int max) {
	if(num > min) {
		if(num < max) {
			return num;
		}
		return max;
	}
	return min;
}

float iou(Bbox b1, Bbox b2) {
	float area1 = (b1.x2 - b1.x1 + 1) * (b1.y2 - b1.y1 + 1);
	float area2 = (b2.x2 - b2.x1 + 1) * (b2.y2 - b2.y1 + 1);

	float new_x1 = max(b1.x1, b2.x1);
	float new_y1 = max(b1.y1, b2.y1);
	float new_x2 = max(b1.x2, b2.x2);
	float new_y2 = max(b1.y2, b2.y2);

	float dif_new_x = (new_x2 - new_x1 + 1);
	float dif_new_y = (new_y2 - new_y1 + 1);

	float intersection = dif_new_x * dif_new_y;
	return ((dif_new_x > 0) && (dif_new_y > 0)) ? intersection / (area2 + area1 - intersection) : 0;
}

vector<Bbox> nms(vector<Bbox> &boxes) {
	sort(boxes.begin(), boxes.end(), [](Bbox b1, Bbox b2){return b1.confidence > b2.confidence;});

	vector<Bbox> picked_Bbox;
	while(boxes.size() > 0) {
		picked_Bbox.emplace_back(boxes[0]);
		boxes.erase(boxes.begin());

		int i=0;
		while(i < (int)boxes.size()) {
			float iou_score = iou(picked_Bbox.back(), boxes[i]);
			if(iou_score >= IOU_THRESHOLD) {
				boxes.erase(boxes.begin());
				continue;
			}
			i++;
		}
	}

	return picked_Bbox;
}

vector<Bbox> box_find(int8_t* outputs) {
	vector<Bbox> boxes;

	for(int i=0;i<CANDIDATE;i++) {
		float confidence = DEQNT(outputs[i*(CLASS_NUM+5) + 4]);
		if(confidence >= CONFI_THRESHOLD) {
			int max_class_id = -1;
			float max_class_score = 0.0;
			for (int j=0;j<CLASS_NUM;j++) {
				float score = DEQNT(outputs[i*(CLASS_NUM+5) + (5+j)]);
				if(score > max_class_score) {
					max_class_id = j;
					max_class_score = score;
				}
			}
			if(max_class_score >= SCORE_THRESHOLD) {
				float x = DEQNT(outputs[i*(CLASS_NUM+5) + 0]);
				float y = DEQNT(outputs[i*(CLASS_NUM+5) + 1]);
				float w = DEQNT(outputs[i*(CLASS_NUM+5) + 2]);
				float h = DEQNT(outputs[i*(CLASS_NUM+5) + 3]);
				int x1 = clamp((x-w / 2) * PIC_SIZE, 0, PIC_SIZE);
				int y1 = clamp((y-h / 2) * PIC_SIZE, 0, PIC_SIZE);
				int x2 = clamp((x+w / 2) * PIC_SIZE, 0, PIC_SIZE);
				int y2 = clamp((y+h / 2) * PIC_SIZE, 0, PIC_SIZE);
				boxes.emplace_back(Bbox(x1, y1, x2, y2, confidence, max_class_id));
			}
		}
	}

	return boxes;
}

vector<Bbox> box_select(vector<Bbox> boxes) {
	int box_size = boxes.size();
	vector<Bbox> show;

	for(int i=0;i<CLASS_NUM;i++) {
		vector<Bbox> class_box;
		for(int j=0;j<box_size;j++) {
			if(boxes[j].class_id == i) {
				class_box.emplace_back(boxes[j]);
			}
		}
		if(class_box.size() > 0) {
			vector<Bbox> result = nms(class_box);
			int result_size = result.size();
			if(result_size > 0) {
				show.insert(show.end(), result.begin(), result.end());
			}
		}
	}

	return show;
}

void post_process(int8_t* outputs) {
	vector<Bbox> boxes = box_find(outputs);

	boxes = box_select(boxes);

	int boxes_size = boxes.size();
	for(int i=0;i<boxes_size;i++) {
		TVMLogf("class: %d, x1:%d, y1:%d, x2:%d, y2:%d, confidence:%f\r\n", boxes[i].class_id, boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, boxes[i].confidence*100);
	}
}
