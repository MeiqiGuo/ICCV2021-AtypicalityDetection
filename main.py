from __future__ import print_function
#%matplotlib inline
import logging
import os
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import random
import json

from config.param import args
from models.svte import MaskVTE, MaskSVTE
from datasets.ads import AdsDatasetWithMask, AtypicalDatasetWithMask
from models.position_encoding import compute_relative_position


os.makedirs(args.output_dir, exist_ok=True)
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=os.path.join(args.output_dir, "log.log"), level=logging.INFO, format=LOG_FORMAT)
logging.info(args)

dataset = {}
dataloader = {}

if args.train:
    for phase in ["train", "val"]:
        dataset[phase] = AdsDatasetWithMask(phase, args.img_folder, args.annotation_path, args.feat_folder)
        logging.info("{0} images are loaded for {1} set.".format(len(dataset[phase]), phase))
        dataloader[phase] = torch.utils.data.DataLoader(dataset[phase], batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.num_workers)
if args.test:
    dataset["test"] = AdsDatasetWithMask("test", args.img_folder, args.annotation_path, args.feat_folder)
    logging.info("{0} images are loaded for test set.".format(len(dataset["test"])))
    dataloader["test"] = torch.utils.data.DataLoader(dataset["test"], batch_size=args.batch_size_eval,
                                             shuffle=False, num_workers=args.num_workers)

if args.atypical_test:
    dataset["atypical"] = AtypicalDatasetWithMask(args.img_folder, args.atypical_test_path, args.feat_folder)
    logging.info("{0} images are loaded for atypical set.".format(len(dataset["atypical"])))
    dataloader["atypical"] = torch.utils.data.DataLoader(dataset["atypical"], batch_size=args.batch_size_eval,
                                                     shuffle=False, num_workers=args.num_workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.vte:
    model = MaskVTE(args.d_model, args.d_feat, args.d_pos, args.n_head, args.n_layer, args.dropout)
elif args.svte:
    model = MaskSVTE(args.d_model, args.d_feat, args.d_pos, args.n_head, args.n_layer, args.dropout)
else:
    logging.error("Please specify a model")

model = model.to(device)

if args.train:

    if args.load_model:
        model_path = args.load_model
        logging.info("Load model from %s" % model_path)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    mse = torch.nn.MSELoss(reduction="none")
    best_valid = 1000
    iter_wrapper = (lambda x: tqdm(x, total=len(dataloader["train"])))
    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()
        for features, boxes in iter_wrapper(dataloader["train"]):
            model.zero_grad()
            features = features.to(device)
            bsz, seq_len, _ = boxes.size()
            mask = torch.tensor([random.sample(range(seq_len), args.n_mask) for tmp in range(bsz)]).to(device)
            if args.vte:
                boxes = boxes.to(device)
                output, target = model(features.transpose(0, 1), boxes.transpose(0, 1), mask.transpose(0, 1))
            elif args.svte:
                pos, pos_dim = compute_relative_position(boxes, pos_dim=args.d_pos, is_cartesian=args.cartesian, is_polar=args.polar, iou=args.iou)
                assert pos_dim == args.d_pos
                pos = torch.Tensor(pos).to(device)
                output, target = model(features.transpose(0, 1), pos, mask.transpose(0, 1))

            loss = mse(output, target).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss * features.size()[0]

        epoch_loss /= dataset["train"].__len__()
        logging.info("************************************************\n")
        logging.info("Epoch %d: Train loss %0.2f\n" % (epoch, epoch_loss))

        # Do Validation
        valid_loss = 0
        model.eval()
        for features, boxes in dataloader["val"]:
            with torch.no_grad():
                features = features.to(device)
                bsz, seq_len, _ = boxes.size()
                mask = torch.tensor([random.sample(range(seq_len), args.n_mask) for tmp in range(bsz)]).to(device)

                if args.vte:
                    boxes = boxes.to(device)
                    output, target = model(features.transpose(0, 1), boxes.transpose(0, 1), mask.transpose(0, 1))
                elif args.svte:
                    pos, pos_dim = compute_relative_position(boxes, pos_dim=args.d_pos, is_cartesian=args.cartesian, is_polar=args.polar, iou=args.iou)
                    assert pos_dim == args.d_pos
                    pos = torch.Tensor(pos).to(device)
                    output, target = model(features.transpose(0, 1), pos, mask.transpose(0, 1))

                loss = mse(output, target).mean()
                valid_loss += loss * features.size()[0]

        valid_loss /= dataset["val"].__len__()

        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "BEST.pth"))

        logging.info("Epoch %d: Valid loss %0.2f; " % (epoch, valid_loss) + "Epoch %d: Best loss %0.2f\n" % (epoch, best_valid))

    torch.save(model.state_dict(), os.path.join(args.output_dir, "LAST.pth"))

if args.test:
    if args.load_model:
        model_path = args.load_model
    else:
        model_path = os.path.join(args.output_dir, "BEST.pth")
    logging.info("Load model from %s" % model_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    mse = torch.nn.MSELoss(reduction="none")
    # Do Validation
    valid_loss = 0
    model.eval()
    for features, boxes in dataloader["test"]:
        with torch.no_grad():
            features = features.to(device)
            bsz, seq_len, _ = boxes.size()
            mask = torch.tensor([random.sample(range(seq_len), args.n_mask) for tmp in range(bsz)])
            if args.vte:
                boxes = boxes.to(device)
                output, target = model(features.transpose(0, 1), boxes.transpose(0, 1), mask.transpose(0, 1))
            elif args.svte:
                pos, pos_dim = compute_relative_position(boxes, pos_dim=args.d_pos, is_cartesian=args.cartesian, is_polar=args.polar, iou=args.iou)
                assert pos_dim == args.d_pos
                pos = torch.Tensor(pos).to(device)
                output, target = model(features.transpose(0, 1), pos, mask.transpose(0, 1))

            loss = mse(output, target).mean()
            valid_loss += loss * features.size()[0]
    valid_loss /= dataset["test"].__len__()
    logging.info("Test loss %0.2f; " % (valid_loss))

if args.atypical_test:
    if args.load_model:
        model_path = args.load_model
    else:
        model_path = os.path.join(args.output_dir, "BEST.pth")
    logging.info("Load model from %s" % model_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    mse = torch.nn.MSELoss(reduction="none")
    valid_loss = 0
    model.eval()
    img_ids = []
    labels = []
    ave_losses = []
    max_losses = []
    max_loss_boxes = []
    top_boxes = []
    total_boxes = []
    total_losses = []
    for img_id, features, boxes, label in dataloader["atypical"]:
        with torch.no_grad():
            features = features.to(device)
            bsz, seq_len, _ = boxes.size()
            batch_loss = torch.zeros(seq_len, bsz)
            if args.vte:
                boxes = boxes.to(device)
                for i in range(seq_len):
                    mask = torch.tensor([[[i]] * bsz]).squeeze(0)
                    output, target = model(features.transpose(0, 1), boxes.transpose(0, 1), mask.transpose(0, 1))
                    loss = mse(output, target).mean(2).mean(0)
                    batch_loss[i] = loss
            elif args.svte:
                pos, pos_dim = compute_relative_position(boxes, pos_dim=args.d_pos, is_cartesian=args.cartesian, is_polar=args.polar, iou=args.iou)
                assert pos_dim == args.d_pos
                pos = torch.Tensor(pos).to(device)
                for i in range(seq_len):
                    mask = torch.tensor([[[i]] * bsz]).squeeze(0)
                    output, target = model(features.transpose(0, 1), pos, mask.transpose(0, 1))
                    loss = mse(output, target).mean(2).mean(0)
                    batch_loss[i] = loss

            ave_loss = batch_loss.mean(0)
            max_loss, max_id = torch.max(batch_loss, dim=0)
            valid_loss += batch_loss.mean(0).sum()
            ave_losses.extend(ave_loss.tolist())
            max_losses.extend(max_loss.tolist())
            img_ids.extend(img_id)
            labels.extend(label.tolist())
            top_boxes.extend(boxes[list(range(bsz)), max_id, :].tolist())
            total_boxes.extend(boxes.tolist())
            total_losses.extend(batch_loss.transpose(0, 1).tolist())

    valid_loss /= dataset["atypical"].__len__()
    logging.info("Test loss %0.2f; " % (valid_loss))

    # compute AUC score for ave loss
    fpr, tpr, _ = roc_curve(labels, ave_losses)
    roc_auc = auc(fpr, tpr)
    print("AUC score with ave loss: {}".format(roc_auc))

    # save a json file with details
    result = {}
    for img_id, ave_loss, max_loss, label, top_box, losses, boxes in list(zip(img_ids, ave_losses, max_losses, labels, top_boxes, total_losses, total_boxes)):
        result[img_id] = {}
        result[img_id]["ave_loss"] = ave_loss
        result[img_id]["max_loss"] = max_loss
        result[img_id]["label"] = label
        result[img_id]["top_box"] = top_box
        result[img_id]["losses"] = losses
        result[img_id]["boxes"] = boxes

    json.dump(result, open(os.path.join(args.output_dir, "pred_results.json"), 'w'))



