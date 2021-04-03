
# Patients

{
  AllPatientsQuery {
    entries
  }
}

mutation ($id: id!) {
  PatientRemove(id: $id) {
    id
  }
}
mutation ($id: id!) {
  PatientRemove(id: $id) {
    id
  }
}

mutation ($id: id, $patient: Patient_Input!) {
  PatientCreate(id: $id resource: $patient) {
    id
  }
}
{
  "id": "e1a1e10a-ee0a-4a95-977c-5e49f6f8abf9",
  "patient": {
    "resourceType": "Patient",
    "name": {
      "text": "Jane Doe"
    }
  }
}

# Encounters

mutation ($id: id, $input: Encounter_Input!) {
  EncounterCreate(id: $id resource: $input) {
    id
  }
}

{
  "id": "ee2b0115-26b6-40aa-8b20-e0fa8140a873",
  "input": {
    "resourceType": "Encounter",
    "status": "finished",
    "class":{
      "code": "AMB",
      "display": "Ambulatory"
    },
    "subject": "899a588e-919f-43cf-9b03-e568087159a5", => Patient Id
    "diagnosis": {
      "id": "b8336c89-59ea-47fa-8287-b1a1ae7d6504",
     "condition": "No finding"
    }
  }
}

query ($patientId: token) {
  EncounterByPatientId(_id: $patientId) {
    entries
  }
}

{
  "patientId": "899a588e-919f-43cf-9b03-e568087159a5"
}

# Media

mutation ($id: id, $input: Media_Input!) {
  MediaCreate(id: $id resource: $input) {
    id
  }
}
{
  "id": "3a26d874-8ffd-43da-af01-78e706526d8c",
  "input": {
    "resourceType": "Media",
    "identifier": {
      "id":"dc09f702-99fa-4a74-bdf0-37008ad49467",
      "value": "Melanoma.jpg"
    },
    "status": "not analyzed",
    "content": {
      "id": "fb695701-6f93-4a65-950a-c3dd9c06f7a7",
      "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Melanoma.jpg/768px-Melanoma.jpg"
    },
    "encounter": "ee2b0115-26b6-40aa-8b20-e0fa8140a873",
    "subject": "899a588e-919f-43cf-9b03-e568087159a5"
  }
}

query ($encounterId: token) {
  MediaByEncounterId(_id: $encounterId) {
    entries
  }
}

{
  "encounterId": "ee2b0115-26b6-40aa-8b20-e0fa8140a873"
}
