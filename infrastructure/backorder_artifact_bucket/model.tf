terraform {
  backend "s3" {
    bucket = "backorder-tf-state"
    key    = "tf_state"
    region = "ap-south-1"
  }
}

resource "aws_s3_bucket" "artifact_bucket" {
  bucket        = "${var.artifacts_bucket_name}"
  force_destroy = var.force_destroy_bucket
}