module.exports = {
  apps: [{
    name: "amazon-review-analyzer",
    script: "./start_app.sh",
    interpreter: "/bin/bash",
    env: {
      NODE_ENV: "production",
    },
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    error_file: "logs/error.log",
    out_file: "logs/out.log",
    merge_logs: true,
    autorestart: true,
    watch: false,
    max_memory_restart: "500M"
  }]
};