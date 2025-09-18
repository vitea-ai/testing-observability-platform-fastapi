{{/*
Expand the name of the chart.
*/}}
{{- define "eval-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "eval-platform.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "eval-platform.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "eval-platform.labels" -}}
helm.sh/chart: {{ include "eval-platform.chart" . }}
{{ include "eval-platform.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "eval-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "eval-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "eval-platform.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "eval-platform.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
API component labels
*/}}
{{- define "eval-platform.apiLabels" -}}
{{ include "eval-platform.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "eval-platform.apiSelectorLabels" -}}
{{ include "eval-platform.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Worker component labels
*/}}
{{- define "eval-platform.workerLabels" -}}
{{ include "eval-platform.labels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Worker selector labels
*/}}
{{- define "eval-platform.workerSelectorLabels" -}}
{{ include "eval-platform.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Evaluator component labels
*/}}
{{- define "eval-platform.evaluatorLabels" -}}
{{ include "eval-platform.labels" . }}
app.kubernetes.io/component: evaluator
{{- end }}

{{/*
Evaluator selector labels
*/}}
{{- define "eval-platform.evaluatorSelectorLabels" -}}
{{ include "eval-platform.selectorLabels" . }}
app.kubernetes.io/component: evaluator
{{- end }}

{{/*
Database URL construction
*/}}
{{- define "eval-platform.databaseUrl" -}}
{{- if .Values.env.APP_DATABASE_URL -}}
{{- .Values.env.APP_DATABASE_URL }}
{{- else -}}
postgresql://{{ .Values.env.DB_USER }}:{{ "{{" }} .Values.externalSecrets.secrets.dbPassword {{ "}}" }}@{{ .Values.env.DB_HOST }}:{{ .Values.env.DB_PORT }}/{{ .Values.env.DB_NAME }}
{{- end }}
{{- end }}