
resource "aws_s3_bucket" "prediction_bucket_name" {
  bucket        = "${var.prediction_bucket_name}"
  force_destroy = var.force_destroy_bucket
}