const { GraphQLString } = require('graphql');
const TokenScalar = require('../scalars/token.scalar.js');
const DateScalar = require('../scalars/date.scalar.js');
const UriScalar = require('../scalars/uri.scalar.js');

/**
 * @name exports
 * @static
 * @summary Arguments for the nutritionorder query
 */
module.exports = {
	// http://hl7.org/fhir/SearchParameter/clinical-identifier
	identifier: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'NutritionOrder.identifier',
		description:
			'Multiple Resources:     * [AllergyIntolerance](allergyintolerance.html): External ids for this item  * [CarePlan](careplan.html): External Ids for this plan  * [CareTeam](careteam.html): External Ids for this team  * [Composition](composition.html): Version-independent identifier for the Composition  * [Condition](condition.html): A unique identifier of the condition record  * [Consent](consent.html): Identifier for this record (external references)  * [DetectedIssue](detectedissue.html): Unique id for the detected issue  * [DeviceRequest](devicerequest.html): Business identifier for request/order  * [DiagnosticReport](diagnosticreport.html): An identifier for the report  * [DocumentManifest](documentmanifest.html): Unique Identifier for the set of documents  * [DocumentReference](documentreference.html): Master Version Specific Identifier  * [Encounter](encounter.html): Identifier(s) by which this encounter is known  * [EpisodeOfCare](episodeofcare.html): Business Identifier(s) relevant for this EpisodeOfCare  * [FamilyMemberHistory](familymemberhistory.html): A search by a record identifier  * [Goal](goal.html): External Ids for this goal  * [ImagingStudy](imagingstudy.html): Identifiers for the Study, such as DICOM Study Instance UID and Accession number  * [Immunization](immunization.html): Business identifier  * [List](list.html): Business identifier  * [MedicationAdministration](medicationadministration.html): Return administrations with this external identifier  * [MedicationDispense](medicationdispense.html): Returns dispenses with this external identifier  * [MedicationRequest](medicationrequest.html): Return prescriptions with this external identifier  * [MedicationStatement](medicationstatement.html): Return statements with this external identifier  * [NutritionOrder](nutritionorder.html): Return nutrition orders with this external identifier  * [Observation](observation.html): The unique id for a particular observation  * [Procedure](procedure.html): A unique identifier for a procedure  * [RiskAssessment](riskassessment.html): Unique identifier for the assessment  * [ServiceRequest](servicerequest.html): Identifiers assigned to this order  * [SupplyDelivery](supplydelivery.html): External identifier  * [SupplyRequest](supplyrequest.html): Business Identifier for SupplyRequest  * [VisionPrescription](visionprescription.html): Return prescriptions with this external identifier  ',
	},
	// http://hl7.org/fhir/SearchParameter/clinical-patient
	patient: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'NutritionOrder.patient',
		description:
			'Multiple Resources:     * [AllergyIntolerance](allergyintolerance.html): Who the sensitivity is for  * [CarePlan](careplan.html): Who the care plan is for  * [CareTeam](careteam.html): Who care team is for  * [ClinicalImpression](clinicalimpression.html): Patient or group assessed  * [Composition](composition.html): Who and/or what the composition is about  * [Condition](condition.html): Who has the condition?  * [Consent](consent.html): Who the consent applies to  * [DetectedIssue](detectedissue.html): Associated patient  * [DeviceRequest](devicerequest.html): Individual the service is ordered for  * [DeviceUseStatement](deviceusestatement.html): Search by subject - a patient  * [DiagnosticReport](diagnosticreport.html): The subject of the report if a patient  * [DocumentManifest](documentmanifest.html): The subject of the set of documents  * [DocumentReference](documentreference.html): Who/what is the subject of the document  * [Encounter](encounter.html): The patient or group present at the encounter  * [EpisodeOfCare](episodeofcare.html): The patient who is the focus of this episode of care  * [FamilyMemberHistory](familymemberhistory.html): The identity of a subject to list family member history items for  * [Flag](flag.html): The identity of a subject to list flags for  * [Goal](goal.html): Who this goal is intended for  * [ImagingStudy](imagingstudy.html): Who the study is about  * [Immunization](immunization.html): The patient for the vaccination record  * [List](list.html): If all resources have the same subject  * [MedicationAdministration](medicationadministration.html): The identity of a patient to list administrations  for  * [MedicationDispense](medicationdispense.html): The identity of a patient to list dispenses  for  * [MedicationRequest](medicationrequest.html): Returns prescriptions for a specific patient  * [MedicationStatement](medicationstatement.html): Returns statements for a specific patient.  * [NutritionOrder](nutritionorder.html): The identity of the person who requires the diet, formula or nutritional supplement  * [Observation](observation.html): The subject that the observation is about (if patient)  * [Procedure](procedure.html): Search by subject - a patient  * [RiskAssessment](riskassessment.html): Who/what does assessment apply to?  * [ServiceRequest](servicerequest.html): Search by subject - a patient  * [SupplyDelivery](supplydelivery.html): Patient for whom the item is supplied  * [VisionPrescription](visionprescription.html): The identity of a patient to list dispenses for  ',
	},
	// http://hl7.org/fhir/SearchParameter/clinical-encounter
	encounter: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'NutritionOrder.encounter',
		description:
			'Multiple Resources:     * [Composition](composition.html): Context of the Composition  * [DeviceRequest](devicerequest.html): Encounter during which request was created  * [DiagnosticReport](diagnosticreport.html): The Encounter when the order was made  * [DocumentReference](documentreference.html): Context of the document  content  * [Flag](flag.html): Alert relevant during encounter  * [List](list.html): Context in which list created  * [NutritionOrder](nutritionorder.html): Return nutrition orders with this encounter identifier  * [Observation](observation.html): Encounter related to the observation  * [Procedure](procedure.html): Encounter created as part of  * [RiskAssessment](riskassessment.html): Where was assessment performed?  * [ServiceRequest](servicerequest.html): An encounter in which this request is made  * [VisionPrescription](visionprescription.html): Return prescriptions with this encounter identifier  ',
	},
	// http://hl7.org/fhir/SearchParameter/NutritionOrder-additive
	additive: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'NutritionOrder.enteralFormula.additiveType',
		description: 'Type of module component to add to the feeding',
	},
	// http://hl7.org/fhir/SearchParameter/NutritionOrder-datetime
	datetime: {
		type: DateScalar,
		fhirtype: 'date',
		xpath: 'NutritionOrder.dateTime',
		description: 'Return nutrition orders requested on this date',
	},
	// http://hl7.org/fhir/SearchParameter/NutritionOrder-formula
	formula: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'NutritionOrder.enteralFormula.baseFormulaType',
		description: 'Type of enteral or infant formula',
	},
	// http://hl7.org/fhir/SearchParameter/NutritionOrder-instantiates-canonical
	instantiates_canonical: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'NutritionOrder.instantiatesCanonical',
		description: 'Instantiates FHIR protocol or definition',
	},
	// http://hl7.org/fhir/SearchParameter/NutritionOrder-instantiates-uri
	instantiates_uri: {
		type: UriScalar,
		fhirtype: 'uri',
		xpath: 'NutritionOrder.instantiatesUri',
		description: 'Instantiates external protocol or definition',
	},
	// http://hl7.org/fhir/SearchParameter/NutritionOrder-oraldiet
	oraldiet: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'NutritionOrder.oralDiet.type',
		description:
			'Type of diet that can be consumed orally (i.e., take via the mouth).',
	},
	// http://hl7.org/fhir/SearchParameter/NutritionOrder-provider
	provider: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'NutritionOrder.orderer',
		description: 'The identity of the provider who placed the nutrition order',
	},
	// http://hl7.org/fhir/SearchParameter/NutritionOrder-status
	status: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'NutritionOrder.status',
		description: 'Status of the nutrition order.',
	},
	// http://hl7.org/fhir/SearchParameter/NutritionOrder-supplement
	supplement: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'NutritionOrder.supplement.type',
		description: 'Type of supplement product requested',
	},
};
